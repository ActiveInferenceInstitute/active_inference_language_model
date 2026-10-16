/**
 * Static generative matrices drawn as raster heatmaps on canvas.
 */

import { store } from './state.js';

async function fetchStaticMatrices() {
    try {
        const res = await fetch('/api/static_matrices');
        store.staticMatrices = await res.json();
        renderMatrixHeatmaps();
    } catch (e) {
        console.warn('Matrix fetch error:', e);
    }
}

function drawHeatmap(canvasId, matrix) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !matrix || !matrix.length) return;
    const ctx = canvas.getContext('2d');
    const rows = matrix.length;
    const cols = matrix[0].length;
    canvas.width = cols;
    canvas.height = rows;

    const imgData = ctx.createImageData(cols, rows);
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const val = Math.min(1.0, Math.max(0.0, matrix[r][c]));
            // Perceptual Viridis-like tint: deep blue to teal to yellow
            const idx = (r * cols + c) * 4;
            imgData.data[idx] = Math.floor(val * 240);       // R
            imgData.data[idx + 1] = Math.floor(val * 220);   // G
            imgData.data[idx + 2] = Math.floor(255 - val * 120); // B
            imgData.data[idx + 3] = 255;                     // Alpha
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

function renderMatrixHeatmaps() {
    if (!store.staticMatrices) return;
    drawHeatmap('matrix-l1-canvas', store.staticMatrices.A_letter);
    drawHeatmap('matrix-g2p-canvas', store.staticMatrices.A_g2p);
    drawHeatmap('matrix-case-canvas', store.staticMatrices.B_case);
}


// Reply length, read defensively. This element was addressed by the
// script and never declared in the markup, so every call to
// runInference() threw a TypeError while building the request body —
// before any fetch, so nothing reached the network tab and the panels
// simply stayed on their placeholders.

export { fetchStaticMatrices, drawHeatmap, renderMatrixHeatmaps };
