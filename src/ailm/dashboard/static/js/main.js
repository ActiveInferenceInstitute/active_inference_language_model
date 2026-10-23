/**
 * Entry point: load configuration, wire lifecycle, start the page.
 *
 * ## Why functions are attached to `window`
 *
 * The markup uses inline `onclick="..."` handlers, which resolve against the
 * global scope. ES modules do not populate it, so every function a handler
 * names is attached here, explicitly and in one list. That is a deliberate
 * trade: inline handlers keep each control's behavior readable in the markup
 * beside the control, and the alternative — a block of `addEventListener` calls
 * far from the elements they bind — moves that information somewhere the reader
 * has to go looking for it.
 *
 * `tests/test_dashboard.py` checks that every inline handler resolves, so a
 * function that stops being exported here fails a test rather than becoming a
 * dead button.
 */

// Load dependencies first, one at a time. Static imports fan out beyond the
// local server's worker ceiling on a cold page and can reset module requests.
// Keep the config namespace: loadFailed is a live binding changed by load().
const config = await import('./config.js');
const { load: loadConfig, applyToControls } = config;
await import('./state.js');
const { setStatus } = await import('./status.js');
const {
    stopSpeaking,
    speak,
    speakCurrentSentence,
    speakTopDownContinuation,
    speakFullUtterance,
} = await import('./speech.js');
const { fetchStaticMatrices } = await import('./matrices.js');
const { runInference, onSliderChange } = await import('./inference.js');
const { exportChat, renderChat, resetChat, sendChat, toggleChatSTT } = await import('./chat.js');
const { toggleSTT } = await import('./stt.js');

export async function showDashboardView(view, reload = false) {
    if (!['inspector', 'explorer'].includes(view)) return;
    stopSpeaking();
    document.querySelector('.header-actions').hidden = view !== 'inspector';
    for (const name of ['inspector', 'explorer']) {
        document.getElementById(`${name}-view`).hidden = name !== view;
        document.getElementById(`${name}-view-button`).setAttribute('aria-pressed', String(name === view));
    }
    if (view !== 'explorer') return;
    const frame = document.getElementById('explorer-frame');
    const status = document.getElementById('explorer-status');
    if (frame.getAttribute('src') && !reload) return;
    frame.hidden = true;
    status.textContent = 'Loading the generated explorer…';
    try {
        const response = await fetch(frame.dataset.src, { cache: 'no-store' });
        if (!response.ok) {
            const payload = await response.json();
            throw new Error(payload.error || `Explorer request failed (${response.status})`);
        }
        frame.src = frame.dataset.src;
        frame.hidden = false;
        status.textContent = 'Generated explorer loaded. Its self-check reports model consistency below.';
    } catch (error) {
        status.textContent = `Could not load the explorer: ${error.message}`;
    }
}

// Inline handlers resolve against the global scope; these are the exact names
// the markup references.
Object.assign(window, {
    showDashboardView,
    runInference,
    onSliderChange,
    sendChat,
    resetChat,
    exportChat,
    toggleChatSTT,
    toggleSTT,
    stopSpeaking,
    speak,
    speakCurrentSentence,
    speakTopDownContinuation,
    speakFullUtterance,
    setStatus,
});

// A backgrounded or closed tab keeps speaking otherwise: `speechSynthesis` is
// per-origin and outlives this page's visibility.
document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopSpeaking();
});
window.addEventListener('pagehide', () => stopSpeaking());
window.addEventListener('beforeunload', () => stopSpeaking());

async function startDashboard() {
    // Configuration first: the controls below take their bounds from it, and
    // starting inference against defaults the server does not share would show
    // a limit the server then clamps.
    await loadConfig();
    if (config.loadFailed) {
        setStatus('Could not read server configuration; inference is disabled.', 'error');
        return;
    }
    applyToControls();
    fetchStaticMatrices();
    runInference();
    renderChat();

}

// Awaited imports may finish after DOMContentLoaded has already fired.
if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', startDashboard, { once: true });
} else {
    startDashboard();
}
