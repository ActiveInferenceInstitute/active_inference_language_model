#!/usr/bin/env python3
"""Assemble the generated standalone AILM local demo. Records nothing.

**This is not the project's main interface.** ``src/ailm/dashboard/`` is: it runs
Levels 1 through 6 with bidirectional chat and speech, and it is where you go to
inspect the model. What this builds is an ignored local artifact — one HTML with
no server or network after generation. It is useful for local exploration and
QA, but it is not tracked or distributed as a publication artifact. It is a
strict subset by design, and
``src/ailm/dashboard/README.md`` carries the comparison.

Both read the same builders out of ``ailm.levels.*``, so they are two views and
not two models; ``tests/test_interface_inventory.py`` fails if that stops being
true, and refuses an undeclared third interface.

The page inlines ``output/web/ailm_model.json`` so it runs from a file:// URL with
no server and no network. Both JSON and HTML are generated and ignored; rerun
the exporter and this builder to reproduce them.

What the page does that a figure cannot: it *runs the model*. The forward filter,
the top-down lexicon preference and the policy posterior are the same arithmetic
the Python build performs, over the same matrices, so a reader can type a string
and watch the recursion rather than read that it happened.

The honesty gate is `selfCheck()` in the page: it recomputes
``stage1_letter_perplexity`` from the inlined transition and compares against the
recorded claim, recomputes the information-gain term of the expected free energy,
and checks fixed-prefix policy posteriors against Python probes at both the
recorded operating precision and gamma 16. If any comparison disagrees the page
renders an error instead of the demo. A demo that silently drifts from the paper
is worse than no demo, because it looks like corroboration.

Writes ``output/web/ailm.html``.

Usage:
    .venv/bin/python scripts/build_web_ailm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _artifact_io import atomic_write_text  # noqa: E402

MODEL = REPO_ROOT / "output" / "web" / "ailm_model.json"
OUTPUT = REPO_ROOT / "output" / "web" / "ailm.html"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AILM — the model, running</title>
<style>
  :root {
    --bg:#fbfaf8; --fg:#14110e; --muted:#6b6357; --rule:#ddd6ca;
    --accent:#8a5a2b; --good:#2f6b46; --bad:#9b3226; --card:#fff;
    --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg:#14120f; --fg:#ece7df; --muted:#9d958a; --rule:#332e28;
      --accent:#d99a5b; --good:#7bbb95; --bad:#e08b7d; --card:#1c1916;
    }
  }
  :root[data-theme="dark"] {
    --bg:#14120f; --fg:#ece7df; --muted:#9d958a; --rule:#332e28;
    --accent:#d99a5b; --good:#7bbb95; --bad:#e08b7d; --card:#1c1916;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--bg); color:var(--fg);
    font:16px/1.6 Georgia,"Iowan Old Style",serif;
    padding:2.5rem 1.25rem 5rem;
  }
  .wrap { max-width:64rem; margin:0 auto; }
  h1 { font-size:1.9rem; line-height:1.2; margin:0 0 .3rem; letter-spacing:-.01em; }
  .sub { color:var(--muted); margin:0 0 2rem; font-size:1.02rem; }
  h2 {
    font-size:1.15rem; margin:2.6rem 0 .6rem; padding-bottom:.35rem;
    border-bottom:1px solid var(--rule); letter-spacing:-.005em;
  }
  p { margin:.7rem 0; }
  .muted { color:var(--muted); }
  .card {
    background:var(--card); border:1px solid var(--rule);
    border-radius:9px; padding:1.1rem 1.2rem; margin:1rem 0;
  }
  code, .mono { font-family:var(--mono); font-size:.88em; }
  input[type=text] {
    width:100%; font-family:var(--mono); font-size:1rem; padding:.6rem .7rem;
    border:1px solid var(--rule); border-radius:6px;
    background:var(--bg); color:var(--fg);
  }
  input[type=range] { width:100%; accent-color:var(--accent); }
  button {
    font:inherit; font-size:.92rem; padding:.45rem .9rem; cursor:pointer;
    border:1px solid var(--rule); border-radius:6px;
    background:var(--card); color:var(--fg);
  }
  button:hover { border-color:var(--accent); color:var(--accent); }
  table { border-collapse:collapse; width:100%; font-size:.9rem; }
  th, td { text-align:left; padding:.35rem .5rem; border-bottom:1px solid var(--rule); }
  th { font-weight:600; color:var(--muted); font-size:.82rem; letter-spacing:.03em; }
  td.num, th.num { text-align:right; font-family:var(--mono); }
  .scroll { overflow-x:auto; }
  .bars { display:grid; gap:3px; }
  .bar { display:grid; grid-template-columns:1.6rem 1fr 4.2rem; align-items:center; gap:.5rem; }
  .bar .sym { font-family:var(--mono); text-align:center; }
  .bar .track { background:var(--rule); border-radius:3px; height:1rem; overflow:hidden; }
  .bar .fill { display:block; background:var(--accent); height:100%; }
  .bar .val { font-family:var(--mono); font-size:.8rem; text-align:right; color:var(--muted); }
  .pass { color:var(--good); font-weight:600; }
  .fail { color:var(--bad); font-weight:600; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
  @media (max-width:52rem) { .grid2 { grid-template-columns:1fr; } }
  .out { font-family:var(--mono); font-size:.95rem; line-height:1.9; word-break:break-word; }
  .lex { color:var(--good); } .nonlex { color:var(--bad); }
  .word-status { color:var(--muted); font-size:.78em; white-space:nowrap; }
  footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--rule);
           color:var(--muted); font-size:.85rem; }
</style>
</head>
<body>
<div class="wrap">

<h1>AILM, running</h1>
<p class="sub">
  The Active Inference Language Model is conditional probability tables and a
  forward filter. Both are below, executing in this page over the exported
  matrices. Nothing here is a recording of a result.
</p>
<p id="offline-guidance" class="sub" style="font-size:.92rem">
  This is a generated local demo: no server, no network after generation, and a
  strict subset. It is ignored by version control and is not a distributed
  publication artifact. The full local dashboard analyzes input across every
  built level with bidirectional chat and speech &mdash;
  <code>python -m ailm.dashboard</code>, then
  <code>http://127.0.0.1:8080/</code>. Only its L3&rarr;L1g policy path generates
  replies; the other level panels are input analyses, not a reply cascade.
</p>
<p id="embedded-guidance" class="sub" hidden>
  Matrix explorer: execute the exported letter and word matrices locally.
  Switch to Analysis &amp; chat above for input analysis and L3&rarr;L1g replies.
  These views share the same model builders; neither claims a joint reply cascade.
</p>

<div id="selfcheck" class="card" role="status" aria-live="polite"
     aria-atomic="true">checking the model against the ledger…</div>

<h2>1 &nbsp;The letter level, filtering</h2>
<p>
  The emission is the identity, so observing a letter reveals the state exactly
  and the posterior is a point mass. The next prediction is then one column of
  the transition. Type anything; the bars are the model's belief about the
  <em>next</em> symbol, and the running figure is perplexity over what you typed.
</p>
<div class="card">
  <label class="muted" for="txt">Text to filter</label>
  <input id="txt" type="text" spellcheck="false" disabled
         value="the quick brown fox jumps over the lazy dog">
  <div id="filter-results" role="status" aria-live="polite" aria-atomic="true"
       style="display:flex;gap:1.5rem;margin:.9rem 0 .4rem;flex-wrap:wrap">
    <div><span class="muted">perplexity</span>
      <strong id="ppl" class="mono">—</strong></div>
    <div><span class="muted">scored symbols</span>
      <strong id="nsym" class="mono">—</strong></div>
    <div><span class="muted">evaluation fixture</span>
      <strong id="pplref" class="mono">—</strong></div>
  </div>
  <div id="bars" class="bars" role="list" aria-label="Next-symbol probabilities"
       style="margin-top:.8rem"></div>
</div>

<h2>2 &nbsp;Generation, and the term that vanishes</h2>
<p>
  Generation is policy selection. The lexicon's Stage 3 posterior supplies a
  preference
  <code>C</code> over the next letter given the prefix so far; the learned bigram
  supplies the habit <code>E</code>. Because the transition is deterministic and
  controllable, the policy-conditioned predictive is a point mass, the
  information-gain term of the expected free energy is identically zero, and
  selection reduces to <code>&sigma;(&gamma;C + ln E)</code>. Move &gamma; and
  select generate to draw a new sample at that precision.
</p>
<div class="card">
  <label class="muted" for="gam">policy precision &gamma; = <span id="gamv"
    class="mono">0.55</span></label>
  <input id="gam" type="range" min="0.05" max="16" step="0.05" value="0.55" disabled>
  <div class="grid2" style="margin-top:.9rem">
    <div>
      <button id="gen" disabled>generate sample</button>
      <div style="margin-top:.6rem">
        <span class="muted">words</span>
        <strong id="nwords" class="mono">—</strong>
        <span class="muted" style="margin-left:1rem">in lexicon</span>
        <strong id="memb" class="mono">—</strong>
        <span class="muted" style="margin-left:1rem">distinct</span>
        <strong id="dist" class="mono">—</strong>
      </div>
    </div>
    <div class="muted" style="font-size:.88rem">
      Recorded operating point &gamma; = <span id="bestg" class="mono"></span>,
      total variation <span id="besttv" class="mono"></span> from the lexicon
      distribution. At &gamma; = <span id="degg" class="mono"></span> the
      shipped lexicon's greedy prefix path collapses onto one word.
    </div>
  </div>
  <div id="out" class="out" role="status" aria-live="polite" aria-atomic="true"
       aria-label="Generated words with lexicon membership" style="margin-top:.9rem"></div>
</div>

<h2>3 &nbsp;Where Level 4's margin actually lives</h2>
<p>
  The case level's admitted margin is an aggregate over three populations whose
  labels have different provenance. Partitioned by which oracle path assigned
  each label, it is carried entirely by the stratum a dependency rule generated,
  is not shown where English marks case morphologically, and reverses on the
  token-level adposition path.
</p>
<div class="card scroll">
  <table id="l4"><thead><tr>
    <th>oracle path</th><th class="num">tokens</th><th class="num">margin (pts)</th>
    <th>reading</th></tr></thead><tbody></tbody></table>
</div>

<footer>
  Every number on this page is either recorded in <code>data/claims.json</code> or
  computed here from the exported matrices. The self-check above recomputes a
  recorded claim from the inlined transition, checks fixed-prefix policy
  posteriors against Python, and fails the page rather than the assertion. Model
  export: <code>scripts/export_web_model.py</code>; page:
  <code>scripts/build_web_ailm.py</code>.
</footer>
</div>

<script id="model" type="application/json">__MODEL_JSON__</script>
<script>
"use strict";
try {
if (typeof window !== "undefined" && window.parent !== window) {
  document.getElementById("offline-guidance").hidden = true;
  document.getElementById("embedded-guidance").hidden = false;
}
const MODEL_TEXT = (typeof document === "undefined")
  ? globalThis.AILM_MODEL_JSON
  : document.getElementById("model").textContent;
if (typeof MODEL_TEXT !== "string") {
  throw new Error("AILM model JSON is unavailable");
}
const M = JSON.parse(MODEL_TEXT);
const SYM = M.alphabet, N = SYM.length, B = M.letter_transition, C = M.claims;
for (const label of [
  "stage1_letter_perplexity", "generation_best_gamma", "generation_best_tv",
  "generation_degenerate_gamma", "generation_shipped_information_gain_nats",
  "generation_max_information_gain_possible_nats", "stage4_path_deprel_tokens",
  "stage4_path_deprel_margin_pts", "stage4_path_morphology_tokens",
  "stage4_path_morphology_margin_pts", "stage4_path_adposition_tokens",
  "stage4_path_adposition_margin_pts",
]) {
  if (!Number.isFinite(C[label])) throw new Error(`claim ${label} must be a finite number`);
}
const IDX = new Map(SYM.map((s, i) => [s, i]));
const LOG_PREFERENCE_FLOOR = -30;
const POLICY_PARITY_ATOL = 2e-6;

// ── The model ──────────────────────────────────────────────────────────────
// One function, used by the demo and by the self-check alike. If these were two
// implementations the check would be testing the check.
function filterText(text) {
  const chars = [...text.toLowerCase()].filter(c => IDX.has(c));
  if (chars.length < 2) return null;
  let prior = new Array(N).fill(1 / N), logsum = 0;
  for (const ch of chars) {
    const i = IDX.get(ch);
    logsum += Math.log(Math.max(prior[i], 1e-10));
    // Identity emission => posterior is a point mass at i, so the next
    // predictive is exactly column i of the transition.
    prior = B.map(row => row[i]);
  }
  return { perplexity: Math.exp(-logsum / chars.length), n: chars.length, prior };
}

// Trie over the exported lexicon, rebuilt here rather than transported. Each node
// holds the summed Stage 3 prior mass that votes for every next symbol, exactly as
// generation.py's _prefix_next_symbol_masses does.
function newTrieNode() {
  return { children: Object.create(null), terminal: false, nextMass: new Array(N).fill(0) };
}
if (!Array.isArray(M.word_prior) || M.word_prior.length !== M.lexicon.length) {
  throw new Error("word_prior must align one-for-one with lexicon");
}
const TRIE = newTrieNode();
let wordPriorTotal = 0;
for (let i = 0; i < M.lexicon.length; i++) {
  const word = M.lexicon[i], weight = M.word_prior[i];
  if (!Number.isFinite(weight) || weight < 0) throw new Error("word_prior is invalid");
  wordPriorTotal += weight;
  let node = TRIE;
  for (let depth = 0; depth <= word.length; depth++) {
    const symbol = depth < word.length ? word[depth] : " ";
    const symbolIndex = IDX.get(symbol);
    if (symbolIndex === undefined) throw new Error(`lexicon symbol ${symbol} is outside alphabet`);
    node.nextMass[symbolIndex] += weight;
    if (depth < word.length) {
      node = (node.children[symbol] ||= newTrieNode());
    } else {
      node.terminal = true;
    }
  }
}
if (Math.abs(wordPriorTotal - 1) > 1e-12) throw new Error("word_prior must sum to one");
function trieNode(prefix) {
  let node = TRIE;
  for (const ch of prefix) { node = node.children[ch]; if (!node) return null; }
  return node;
}

// The top-down channel: log-preference over the next letter given the prefix.
// Each child receives the normalized sum of the prior weights of the vocabulary
// words that continue through it; space receives the weight of words ending here.
function preference(prefix) {
  const node = trieNode(prefix);
  if (!node) return new Array(N).fill(0);
  const total = node.nextMass.reduce((a, b) => a + b, 0);
  if (!(total > 0)) return new Array(N).fill(0);
  return node.nextMass.map((mass) =>
    Math.max(mass > 0 ? Math.log(mass / total) : LOG_PREFERENCE_FLOOR,
             LOG_PREFERENCE_FLOOR));
}

// q(pi) = softmax(gamma * C + ln E). No information-gain term: with a
// deterministic controllable transition the predictive is a point mass, so the
// expected KL between posterior-after-observation and prior is exactly zero.
function policyPosterior(prefix, currentLetter, gamma) {
  if (!Number.isFinite(gamma) || gamma <= 0) throw new Error("gamma must be positive");
  const currentIndex = IDX.get(currentLetter);
  if (currentIndex === undefined) throw new Error(`unknown current letter ${currentLetter}`);
  const pref = preference(prefix);
  const prefMax = Math.max(...pref);
  const habit = B.map(row => row[currentIndex]);
  const score = pref.map((c, i) =>
    gamma * (c - prefMax) + (habit[i] > 0 ? Math.log(habit[i]) : -Infinity));
  const hi = Math.max(...score.filter(Number.isFinite));
  const ex = score.map(s => Number.isFinite(s) ? Math.exp(s - hi) : 0);
  const z = ex.reduce((a, b) => a + b, 0);
  if (!(z > 0)) throw new Error("policy posterior has zero mass");
  return ex.map(e => e / z);
}

function generationParity() {
  const rows = Array.isArray(M.policy_parity_probes) ? M.policy_parity_probes : [];
  const results = rows.map((probe) => {
    const got = policyPosterior(probe.prefix, probe.current_letter, probe.gamma);
    const expected = probe.posterior;
    const maxAbs = Array.isArray(expected) && expected.length === N
      ? Math.max(...got.map((value, i) => Math.abs(value - expected[i])))
      : Infinity;
    return { prefix: probe.prefix, gamma: probe.gamma, maxAbs };
  });
  const maxAbs = results.length ? Math.max(...results.map(row => row.maxAbs)) : Infinity;
  return { ok: results.length > 0 && maxAbs <= POLICY_PARITY_ATOL, maxAbs, results };
}

function greedyWord(gamma) {
  let prefix = "", current = " ";
  for (let step = 0; step < 20; step++) {
    const posterior = policyPosterior(prefix, current, gamma);
    let pick = 0;
    for (let i = 1; i < posterior.length; i++) if (posterior[i] > posterior[pick]) pick = i;
    const symbol = SYM[pick];
    if (symbol === " " || !/[a-z]/.test(symbol)) break;
    prefix += symbol;
    current = symbol;
  }
  return prefix;
}

function wordPathProbability(word, gamma) {
  let prefix = "", current = " ", probability = 1;
  for (const symbol of word) {
    probability *= policyPosterior(prefix, current, gamma)[IDX.get(symbol)];
    prefix += symbol;
    current = symbol;
  }
  probability *= policyPosterior(prefix, current, gamma)[IDX.get(" ")];
  return probability;
}

// The information-gain term of the expected free energy, from a predictive over
// states under an identity emission:
//
//   IG = sum_o q(o) * KL( q(s|o) || q(s) ),   q(o) = sum_s A[o,s] q(s)
//
// The predictive is an argument so one code path serves both the shipped case and
// its control. That is load-bearing: assigning the posterior and the prior the
// same expression makes log(post/pri) into log(1), and the function then returns
// zero for every input and agrees with a recorded zero always. A check that
// cannot fail is the defect class this project's gate audit exists to catch, and
// it is worse inside a page whose contract is to refuse to render on drift.
function informationGain(predictive) {
  const mass = predictive.reduce((a, b) => a + b, 0);
  const prior = predictive.map((p) => p / mass);
  let gain = 0;
  for (let o = 0; o < N; o++) {
    // Identity emission: A[o, s] is 1 exactly when s === o, so q(o) = prior[o]
    // and the posterior after observing o is a point mass at o, whose KL against
    // the prior is -log(prior[o]).
    const evidence = prior[o];
    if (evidence > 0) gain += evidence * Math.log(1 / evidence);
  }
  return gain;
}

function sample(dist) {
  let r = Math.random(), acc = 0;
  for (let i = 0; i < dist.length; i++) { acc += dist[i]; if (r < acc) return i; }
  return dist.length - 1;
}

function generateWords(gamma, count) {
  const words = [];
  for (let w = 0; w < count; w++) {
    let prefix = "", current = " ";
    for (let step = 0; step < 20; step++) {
      const q = policyPosterior(prefix, current, gamma);
      const pick = SYM[sample(q)];
      if (pick === " " || !/[a-z]/.test(pick)) break;
      prefix += pick; current = pick;
      if (!trieNode(prefix)) break;
    }
    if (prefix) words.push(prefix);
  }
  return words;
}

// ── The honesty gate ───────────────────────────────────────────────────────
function modelDiagnostics() {
  const got = filterText(M.held_out_text);
  if (!got) throw new Error("evaluation fixture must contain at least two supported symbols");
  const want = C.stage1_letter_perplexity;
  const dPpl = Math.abs(got.perplexity - want);
  // Two calls, and the second is the control. A point-mass predictive must give
  // exactly the recorded zero; a uniform one must give the recorded maximum,
  // ln(30). If the control does not fire, the zero means nothing and the page
  // must say so rather than display it.
  const pointMass = new Array(N).fill(0);
  pointMass[0] = 1;
  const ig = informationGain(pointMass);
  const igWant = C.generation_shipped_information_gain_nats;
  const igMax = informationGain(new Array(N).fill(1 / N));
  const igMaxWant = C.generation_max_information_gain_possible_nats;
  const controlFires = Math.abs(igMax - igMaxWant) < 1e-4;
  const parity = generationParity();
  const operating = policyPosterior("", " ", C.generation_best_gamma);
  const degenerate = policyPosterior("", " ", 16);
  const gammaEffect = Math.max(...operating.map((value, i) => Math.abs(value - degenerate[i])));
  const highGammaWord = greedyWord(16);
  const highGammaProbability = wordPathProbability(highGammaWord, 16);
  const highGammaCollapse = highGammaWord.length > 0 && highGammaProbability > 0.99;
  const ok = dPpl < 5e-3 && Math.abs(ig - igWant) < 1e-9 && controlFires
    && parity.ok && gammaEffect > 1e-3 && highGammaCollapse;
  return {
    ok, got, want, ig, igWant, parity, gammaEffect, highGammaWord,
    highGammaProbability, highGammaCollapse,
  };
}

function selfCheck() {
  const el = document.getElementById("selfcheck");
  const report = modelDiagnostics();
  const { got, want, ig, igWant, parity } = report;
  const ok = report.ok;
  el.innerHTML = ok
    ? `<span class="pass">self-check passed.</span> This page recomputed
       <code>stage1_letter_perplexity</code> from the inlined transition and got
       <strong class="mono">${got.perplexity.toFixed(4)}</strong> against the
       recorded <strong class="mono">${want}</strong>, over
       ${got.n.toLocaleString()} evaluation-fixture symbols. The information-gain term of
       the expected free energy computed here is
       <strong class="mono">${ig.toFixed(4)}</strong> nats, matching the recorded
       <strong class="mono">${igWant}</strong>. All fixed-prefix policy posteriors
       match Python within <strong class="mono">${parity.maxAbs.toExponential(2)}</strong>;
       at &gamma; = 16 the greedy word is
       <strong class="mono">${report.highGammaWord}</strong> with path probability
       <strong class="mono">${report.highGammaProbability.toFixed(4)}</strong>. The
       demo below is the same model the paper reports.`
    : `<span class="fail">self-check FAILED.</span> Recomputed perplexity
       ${got.perplexity.toFixed(4)} against recorded ${want}; fixed-prefix policy
       delta ${parity.maxAbs}. The page and the Python model disagree, so nothing
       below should be trusted.`;
  return ok;
}

// ── Rendering ──────────────────────────────────────────────────────────────
function renderBars(prior) {
  const order = prior.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 8);
  document.getElementById("bars").innerHTML = order.map(([p, i]) => {
    const symbol = SYM[i] === " " ? "space" : SYM[i];
    return `
    <div class="bar" role="listitem"
         aria-label="Next symbol ${symbol}, probability ${p.toFixed(4)}">
      <span class="sym">${SYM[i] === " " ? "␣" : SYM[i]}</span>
      <span class="track" aria-hidden="true"><span class="fill" style="width:${(p * 100).toFixed(1)}%"></span></span>
      <span class="val">${p.toFixed(4)}</span>
    </div>`;
  }).join("");
}

function updateFilter() {
  const r = filterText(document.getElementById("txt").value);
  if (!r) {
    document.getElementById("ppl").textContent = "—";
    document.getElementById("nsym").textContent = "0";
    document.getElementById("bars").replaceChildren();
    return;
  }
  document.getElementById("ppl").textContent = r.perplexity.toFixed(4);
  document.getElementById("nsym").textContent = r.n;
  renderBars(r.prior);
}

function updateGeneration() {
  const gamma = parseFloat(document.getElementById("gam").value);
  document.getElementById("gamv").textContent = gamma.toFixed(2);
  const words = generateWords(gamma, 40);
  document.getElementById("nwords").textContent = words.length;
  const inLex = words.filter(w => (trieNode(w) || {}).terminal).length;
  document.getElementById("memb").textContent =
    words.length ? (100 * inLex / words.length).toFixed(1) + "%" : "—";
  document.getElementById("dist").textContent = new Set(words).size;
  document.getElementById("out").innerHTML = words.map(w => {
    const inLexicon = Boolean((trieNode(w) || {}).terminal);
    const status = inLexicon ? "lexicon" : "outside lexicon";
    return `<span class="${inLexicon ? "lex" : "nonlex"}">${w} `
      + `<span class="word-status">[${status}]</span></span>`;
  }).join(" ");
}

function renderL4() {
  const rows = [
    ["dependency relation", C.stage4_path_deprel_tokens, C.stage4_path_deprel_margin_pts,
     "the label is a deterministic function of a syntactic relation"],
    ["morphology (control)", C.stage4_path_morphology_tokens, C.stage4_path_morphology_margin_pts,
     "real English case marking; effect not shown"],
    ["adposition", C.stage4_path_adposition_tokens, C.stage4_path_adposition_margin_pts,
     "label sits on the preposition, not the nominal; reverses"],
  ];
  document.querySelector("#l4 tbody").innerHTML = rows.map(([p, n, m, why]) => `
    <tr><td>${p}</td><td class="num">${n}</td>
    <td class="num" style="color:${m > 5 ? "var(--accent)" : m < 0 ? "var(--bad)" : "inherit"}">
      ${m > 0 ? "+" : ""}${m}</td><td class="muted">${why}</td></tr>`).join("");
}

globalThis.AILMWebModel = Object.freeze({
  generationParity, greedyWord, modelDiagnostics, policyPosterior,
  preference, wordPathProbability,
});

if (typeof document !== "undefined" && selfCheck()) {
  document.getElementById("pplref").textContent = C.stage1_letter_perplexity;
  document.getElementById("bestg").textContent = C.generation_best_gamma;
  document.getElementById("besttv").textContent = C.generation_best_tv;
  document.getElementById("degg").textContent = C.generation_degenerate_gamma;
  document.getElementById("txt").addEventListener("input", updateFilter);
  document.getElementById("gam").addEventListener("input", () =>
    document.getElementById("gamv").textContent =
      parseFloat(document.getElementById("gam").value).toFixed(2));
  document.getElementById("gen").addEventListener("click", updateGeneration);

  updateFilter();
  renderL4();
  updateGeneration();
  for (const id of ["txt", "gam", "gen"]) {
    document.getElementById(id).disabled = false;
  }
}
} catch (error) {
  if (typeof document !== "undefined") {
    const status = document.getElementById("selfcheck");
    status.classList.add("fail");
    status.textContent = "self-check FAILED. The model could not initialize: " + error.message;
  }
  throw error;
}
</script>
</body>
</html>
"""


def main() -> int:
    if not MODEL.exists():
        raise SystemExit(
            f"build refused: {MODEL.relative_to(REPO_ROOT)} is missing. "
            f"Run scripts/export_web_model.py first."
        )
    model = MODEL.read_text(encoding="utf-8")
    # Guard the inline: a closing script tag inside JSON would end the block early
    # and the page would render as text. Cheap to check, silent if it happens.
    if "</script" in model.lower():
        raise SystemExit("build refused: model JSON contains a closing script tag.")
    json.loads(model)  # refuse to inline anything that is not valid JSON

    atomic_write_text(OUTPUT, PAGE.replace("__MODEL_JSON__", model))
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    print("  generated local demo: ignored, no network or server, opens from file://")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
