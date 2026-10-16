/**
 * Speech output: exactly one utterance at a time, always terminated.

`speechSynthesis` is a QUEUE, not a channel. Every speak() appends and cancel()
clears only what is pending at that instant, so two turns in quick succession —
or one turn speaking from more than one path — leave the queue draining after
the turn is over: the interface looks idle and the machine is still talking.
This module is the only thing permitted to queue an utterance, so there is one
place that knows whether anything is in flight.
 */

import { store } from './state.js';

import { cfg } from './config.js';
import { setStatus } from './status.js';

const TTS_UNAVAILABLE = 'Speech synthesis is unavailable in this browser — text output only.';

// Single speech path, so every caller degrades the same way. Three
// separate copies previously disagreed: only one told the user anything
// when the Web Speech API was missing, and the other two failed silently.
// ── Speech output: exactly one utterance at a time, always terminated ──
//
// `speechSynthesis` is a QUEUE, not a channel. Every speak() appends,
// and cancel() only clears what is pending at that instant. Two turns
// in quick succession, or one turn that speaks from more than one code
// path, leave the queue draining after the turn is over: the interface
// looks idle and the machine is still talking. The controller below is
// the only thing in this page allowed to call speak(), so there is one
// place that knows whether anything is in flight.
let activeUtterance = null;
let speechWatchdog = null;

//: A rollout can be long, and a long utterance is the one most likely to
//: outlive the turn that started it. Bounded rather than truncated
//: silently: the reader still sees the full text on screen.
// Bounds come from the server; see js/config.js.

//: Chrome keeps `speaking` true after an utterance ends if the queue was
//: cancelled mid-flight, so the watchdog force-clears rather than trusting it.


function stopSpeaking(reason) {
    if (speechWatchdog) { clearTimeout(speechWatchdog); speechWatchdog = null; }
    activeUtterance = null;
    if (!('speechSynthesis' in window)) return;
    try {
        // cancel() twice with the queue emptied between is the documented
        // way to clear a Chrome queue that has a paused item on it; a
        // single call can leave one utterance to resume later.
        window.speechSynthesis.cancel();
        if (window.speechSynthesis.paused) window.speechSynthesis.resume();
        window.speechSynthesis.cancel();
    } catch (e) {
        console.warn('speech cancel failed', e);
    }
    updateSpeakingIndicator(false);
    if (reason) setStatus(reason);
}

function updateSpeakingIndicator(speaking) {
    const btn = document.getElementById('stop-speaking-btn');
    if (btn) btn.disabled = !speaking;
    const dot = document.getElementById('speaking-indicator');
    if (dot) {
        dot.style.visibility = speaking ? 'visible' : 'hidden';
        dot.setAttribute('aria-label', `Speech status: ${speaking ? 'speaking' : 'idle'}`);
    }
}

function speak(text, rate) {
    // Whatever was saying anything stops first, unconditionally. A new
    // turn never layers on top of the previous one.
    stopSpeaking();

    if (!text) { setStatus('Nothing to speak yet — run inference first.'); return false; }
    if (!('speechSynthesis' in window) || typeof SpeechSynthesisUtterance === 'undefined') {
        setStatus(TTS_UNAVAILABLE, 'error');
        return false;
    }

    const spoken = text.length > cfg.max_spoken_chars
        ? text.slice(0, cfg.max_spoken_chars) + '\u2026'
        : text;

    try {
        const utter = new SpeechSynthesisUtterance(spoken);
        utter.rate = rate;
        const finish = () => {
            if (activeUtterance !== utter) return;  // superseded already
            activeUtterance = null;
            if (speechWatchdog) { clearTimeout(speechWatchdog); speechWatchdog = null; }
            updateSpeakingIndicator(false);
        };
        utter.onend = finish;
        utter.onerror = (e) => {
            finish();
            // `interrupted` and `canceled` are what our own stopSpeaking
            // produces; reporting them as errors would make every normal
            // turn look like a failure.
            if (e.error && e.error !== 'interrupted' && e.error !== 'canceled') {
                setStatus('Speech synthesis error: ' + e.error, 'error');
            }
        };
        activeUtterance = utter;
        updateSpeakingIndicator(true);
        window.speechSynthesis.speak(utter);
        speechWatchdog = setTimeout(() => stopSpeaking(), cfg.speech_watchdog_ms);
        return true;
    } catch (e) {
        stopSpeaking();
        setStatus('Speech synthesis error: ' + e.message, 'error');
        return false;
    }
}

function speakCurrentSentence() {
    speak(document.getElementById('sentence-input').value, 1.0);
}

function speakTopDownContinuation() {
    const c = store.currentInference && store.currentInference.topdown_continuation;
    speak(c && c.continuation_text, 0.95);
}

// Previously read store.currentInference.rollout, a key the engine has never
// emitted, so the "Rollout Audio" button was a permanent no-op. It now
// speaks the full input-plus-continuation utterance, which is the one
// thing the other two buttons do not cover.
function speakFullUtterance() {
    const c = store.currentInference && store.currentInference.topdown_continuation;
    speak(c && c.full_text, 0.95);
}

export { stopSpeaking, updateSpeakingIndicator, speak, speakCurrentSentence, speakTopDownContinuation, speakFullUtterance, TTS_UNAVAILABLE };
