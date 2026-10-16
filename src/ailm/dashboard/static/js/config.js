/**
 * Runtime configuration, fetched from the server that enforces it.

Every bound the interface honors is served by `GET /api/config` from the same
`DashboardConfig` the server validates requests against. Nothing here is a
transcribed copy: a limit written twice is a limit that will disagree with
itself, and the browser copy is the one nobody tests.

`load()` must be awaited before anything reads `cfg`. A failed fetch leaves the
map empty and the entry point stops before issuing inference requests. Guessing
defaults would recreate the exact split-brain configuration this module removes.
 */

export const cfg = {};

export let loadFailed = false;

export async function load() {
    try {
        const response = await fetch('/api/config');
        if (!response.ok) throw new Error('HTTP ' + response.status);
        Object.assign(cfg, await response.json());
        loadFailed = false;
    } catch (e) {
        // A page that cannot read its limits still works; it just uses the
        // shipped defaults. Recorded rather than swallowed so the interface can
        // report that its bounds may not match the server's.
        loadFailed = true;
        console.error('config fetch failed; dashboard inference is disabled', e);
    }
    return cfg;
}

/** Apply server-provided bounds to the inputs that expose them. */
export function applyToControls() {
    const reply = document.getElementById('rollout-len-input');
    if (reply) {
        reply.max = String(cfg.max_continuation_words);
        reply.value = String(cfg.default_reply_words);
    }
    const alpha = document.getElementById('narrative-alpha-input');
    if (alpha) {
        alpha.value = String(cfg.default_narrative_alpha);
        const readout = document.getElementById('alpha-readout');
        if (readout) readout.textContent = Number(alpha.value).toFixed(2);
    }
}
