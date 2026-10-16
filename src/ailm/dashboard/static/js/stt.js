/**
 * Speech recognition for the inspector input.
 *
 * Kept separate from the chat microphone so the two recognition objects cannot
 * fight over one engine: `start()` on an already-running recognizer throws
 * `InvalidStateError`, and a shared object would make one control's stop cancel
 * the other's listen.
 */

import { setStatus } from './status.js';
import { stopSpeaking } from './speech.js';
import { runInference } from './inference.js';
import { speakTopDownContinuation } from './speech.js';

let recognition = null;
let isRecording = false;

if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRec();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
        const text = event.results[0][0].transcript;
        document.getElementById('sentence-input').value = text;
        // Previously called speakDialogicResponse(), which has never
        // been defined: every voice input to the inspector raised a
        // ReferenceError inside an unhandled promise rejection, so the
        // failure was invisible unless the console was open.
        runInference().then(() => speakTopDownContinuation());
    };

    recognition.onend = () => {
        isRecording = false;
        document.getElementById('mic-btn').classList.remove('recording');
    };

    recognition.onerror = (e) => {
        console.error('STT error', e);
        isRecording = false;
        document.getElementById('mic-btn').classList.remove('recording');
        setStatus('Speech recognition error: ' + (e.error || 'unknown'), 'error');
    };
}

function toggleSTT() {
    if (!recognition) {
        setStatus('Speech recognition is unavailable in this browser (try Chrome, Edge, or Safari) — type input instead.', 'error');
        return;
    }
    if (!isRecording) {
        // start() throws InvalidStateError if the engine is still
        // winding down from a previous session; without this the mic
        // button latches into a stuck "recording" state.
        try {
            stopSpeaking();  // never listen while speaking: the page would hear itself
            recognition.start();
            isRecording = true;
            document.getElementById('mic-btn').classList.add('recording');
        } catch (e) {
            isRecording = false;
            document.getElementById('mic-btn').classList.remove('recording');
            setStatus('Could not start the microphone: ' + e.message, 'error');
        }
    } else {
        recognition.stop();
    }
}

// Non-blocking status line. alert() is a modal: it freezes the page and
// stalls any automated verification driving the dashboard.

export { toggleSTT };
