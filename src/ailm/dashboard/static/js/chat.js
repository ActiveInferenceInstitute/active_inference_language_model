/**
 * The bidirectional transcript.

State is client-side and the server is stateless: the transcript is replayed on
every request. That keeps the narrative phase honest — it is a function of the
turn index — without the server holding per-visitor state it would have to
expire.
 */

import { cfg } from './config.js';
import { setStatus } from './status.js';
import { store } from './state.js';
import { speak, stopSpeaking } from './speech.js';
import { setupTimeline, replyLength } from './inference.js';

// ── Bidirectional chat ────────────────────────────────────────────
// The transcript is client-side state; the server is stateless and
// replays the history it is handed on every turn. That keeps the
// narrative phase honest (it is a function of the turn index) without
// the server holding per-visitor state it would have to expire.


let chatRecognition = null;
let chatRecording = false;

function renderChat() {
    const log = document.getElementById('chat-log');
    document.getElementById('chat-turn-count').textContent =
        store.chatHistory.filter(t => t.role === 'user').length;
    if (!store.chatHistory.length) {
        log.innerHTML = '<div class="chat-empty">No turns yet. Say something, or type below.</div>';
        return;
    }
    log.innerHTML = '';
    store.chatHistory.forEach(turn => {
        const wrap = document.createElement('div');
        wrap.className = 'chat-turn ' + turn.role;
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble';
        bubble.textContent = turn.text;
        wrap.appendChild(bubble);
        if (turn.meta) {
            const meta = document.createElement('div');
            meta.className = 'chat-meta';
            meta.textContent = turn.meta;
            wrap.appendChild(meta);
        }
        log.appendChild(wrap);
    });
    log.scrollTop = log.scrollHeight;
}

function resetChat() {
    store.chatHistory = [];
    renderChat();
    setStatus('Conversation cleared. The document-position tercile restarts at "opening".');
}

export function replySpeechAllowed(inspectorHidden, documentHidden, speakReplies) {
    return !inspectorHidden && !documentHidden && speakReplies;
}

async function sendChat() {
    if (store.chatBusy) return;
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) { setStatus('Type a message first.'); return; }

    // Silence the previous turn before starting this one. A reply still
    // draining the queue talks over the next exchange, which is what
    // "it never stops" looks like from outside.
    stopSpeaking();

    store.chatBusy = true;
    document.getElementById('chat-send-btn').disabled = true;
    store.chatHistory.push({ role: 'user', text: message });
    input.value = '';
    renderChat();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                // The user turn just pushed is the CURRENT one, so it is
                // excluded: the server counts prior turns to place this
                // one in the conversation.
                history: store.chatHistory.slice(0, -1).map(t => ({ role: t.role, text: t.text })),
                n_words: replyLength(),
                narrative_alpha: parseFloat(document.getElementById('narrative-alpha-input').value)
            })
        });
        const data = await response.json();
        if (!response.ok || data.error) {
            setStatus('Chat failed: ' + (data.error || response.status), 'error');
            store.chatHistory.pop();
            renderChat();
            return;
        }

        const read = data.read_as || {};
        const surface = (read.surface || []).join(' \u2192 ') || 'no case-bearing token';
        store.chatHistory.push({
            role: 'model',
            text: data.reply,
            meta: `read as ${read.clause_type} (${(read.clause_confidence * 100).toFixed(1)}%) · `
                + `phase ${read.narrative_phase} · turn ${data.turn_index} · `
                + `${read.in_lexicon}/${read.tokens} in lexicon · ${surface}`
        });
        renderChat();

        // The analysis panels follow the conversation. They describe this
        // input turn but, except for the separate L3→L1g rollout, do not
        // causally produce the generated reply.
        if (data.hierarchy) {
            store.currentInference = data.hierarchy;
            setupTimeline(data.hierarchy);
            const inEl = document.getElementById('input-echo');
            const genEl = document.getElementById('generated-echo');
            if (inEl) inEl.textContent = message + ' ';
            if (genEl) genEl.textContent = data.reply;
            const sentenceInput = document.getElementById('sentence-input');
            if (sentenceInput) sentenceInput.value = message;
        }

        // A view switch or backgrounding can occur while /api/chat is in flight.
        // Check attention at speech initiation, after the response, not at send time.
        if (replySpeechAllowed(
            document.getElementById('inspector-view').hidden,
            document.hidden,
            document.getElementById('chat-speak-replies').checked,
        )) {
            speak(data.reply, 0.95);
        }
        setStatus('');
    } catch (e) {
        setStatus('Chat error: ' + e.message, 'error');
        store.chatHistory.pop();
        renderChat();
    } finally {
        store.chatBusy = false;
        document.getElementById('chat-send-btn').disabled = false;
        if (!document.getElementById('inspector-view').hidden) input.focus();
    }
}

// Speech input for the chat box, kept separate from the inspector's mic
// so the two cannot fight over one recognition object.
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const ChatRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    chatRecognition = new ChatRec();
    chatRecognition.continuous = false;
    chatRecognition.interimResults = false;
    chatRecognition.lang = 'en-US';
    chatRecognition.onresult = (event) => {
        document.getElementById('chat-input').value = event.results[0][0].transcript;
        sendChat();
    };
    chatRecognition.onend = () => {
        chatRecording = false;
        document.getElementById('chat-mic-btn').classList.remove('recording');
    };
    chatRecognition.onerror = (e) => {
        chatRecording = false;
        document.getElementById('chat-mic-btn').classList.remove('recording');
        setStatus('Speech recognition error: ' + (e.error || 'unknown'), 'error');
    };
}

function toggleChatSTT() {
    if (!chatRecognition) {
        setStatus('Speech recognition is unavailable in this browser — type instead.', 'error');
        return;
    }
    if (chatRecording) { chatRecognition.stop(); return; }
    try {
        stopSpeaking();  // never listen while speaking: the page would hear itself
        chatRecognition.start();
        chatRecording = true;
        document.getElementById('chat-mic-btn').classList.add('recording');
    } catch (e) {
        chatRecording = false;
        document.getElementById('chat-mic-btn').classList.remove('recording');
        setStatus('Could not start the microphone: ' + e.message, 'error');
    }
}

export { renderChat, resetChat, sendChat, toggleChatSTT };
