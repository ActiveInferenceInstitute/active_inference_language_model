/**
 * The one status line. Non-blocking: `alert()` is a modal that freezes the page and stalls any automation driving it.
 */

function setStatus(msg, kind) {
    const el = document.getElementById('status-line');
    if (!el) return;
    el.textContent = msg || '';
    el.style.color = kind === 'error' ? 'var(--accent-orange)' : 'var(--text-muted)';
    if (msg) { clearTimeout(el._t); el._t = setTimeout(() => { el.textContent = ''; }, 6000); }
}

export { setStatus };
