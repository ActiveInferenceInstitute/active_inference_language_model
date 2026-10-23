/**
 * Driving /api/infer and rendering every level panel from one payload.

The timeline slider and the level cards are two views of the same belief
trajectory, so they are rendered together: a slider position that disagreed with
the panels would be a display bug indistinguishable from a model bug.
 */

import { cfg } from './config.js';
import { setStatus } from './status.js';
import { store } from './state.js';
import { speak, speakTopDownContinuation } from './speech.js';

function replyLength() {
    const el = document.getElementById('rollout-len-input');
    const parsed = parseInt(el && el.value, 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 8;
}

function describeProbabilityBar(row, context, label, probability) {
    row.setAttribute('role', 'listitem');
    row.setAttribute(
        'aria-label',
        `${context}: ${label}, ${(probability * 100).toFixed(2)} percent`
    );
    const track = row.querySelector('.bar-track');
    if (track) track.setAttribute('aria-hidden', 'true');
}

async function runInference() {
    const text = document.getElementById('sentence-input').value.trim();
    if (!text) { setStatus('Enter a sentence to filter.'); return; }

    try {
        const response = await fetch('/api/infer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text,
                n_words: replyLength(),
                narrative_phase: parseInt(document.getElementById('narrative-phase-input').value, 10),
                narrative_alpha: parseFloat(document.getElementById('narrative-alpha-input').value)
            })
        });
        const data = await response.json();
        // The API reports failure as a JSON {error}. Assigning that to
        // store.currentInference and rendering it threw deep inside the
        // timeline setup, so a bad request surfaced as an unrelated
        // stack trace instead of a message.
        if (!response.ok || data.error) {
            setStatus('Inference failed: ' + (data.error || response.status), 'error');
            return;
        }
        setStatus('');
        store.currentInference = data;
        setupTimeline(data);
        const contObj = data.topdown_continuation;
        const inEl = document.getElementById('input-echo');
        const genEl = document.getElementById('generated-echo');
        if (inEl) inEl.textContent = data.text + (data.text.endsWith(' ') ? '' : ' ');
        if (genEl) genEl.textContent = contObj ? contObj.continuation_text : '...';
    } catch (err) {
        // Reported in the status line, not an alert(). A modal blocks the
        // page's own event loop until someone dismisses it, which stops
        // every other control including the one that would retry — and it
        // wedges any automation driving the page.
        console.error('Inference error:', err);
        setStatus('Inference call failed. Is the backend server running? ' + err, 'error');
    }
}

function setupTimeline(data) {
    const steps = data.levels.l1_letter.steps;
    const slider = document.getElementById('time-slider');
    slider.max = Math.max(0, steps.length - 1);
    slider.value = 0;

    const tracker = document.getElementById('char-tracker');
    tracker.innerHTML = '';
    steps.forEach((s, idx) => {
        const pill = document.createElement('button');
        pill.type = 'button';
        pill.className = 'char-pill' + (idx === 0 ? ' active' : '');
        pill.textContent = s.char === ' ' ? '␣' : s.char;
        pill.setAttribute(
            'aria-label',
            `Show analysis step ${idx + 1}, character ${s.char === ' ' ? 'space' : s.char}`
        );
        if (idx === 0) pill.setAttribute('aria-current', 'step');
        pill.onclick = () => {
            slider.value = idx;
            onSliderChange(idx);
        };
        tracker.appendChild(pill);
    });

    // Render step 0 here rather than leaving it to the caller. It was the
    // caller's job, and the chat path did not do it: a turn rebuilt the
    // timeline while every level panel kept the previous turn's belief state,
    // so the page showed one sentence's characters above another sentence's
    // posteriors. The disclaimer above the chat promises the panels follow the
    // conversation, which made it a claim the page did not keep.
    //
    // A timeline that is not showing a step is not in a valid state, so
    // building one and rendering its first step is one operation.
    onSliderChange(0);
}

function onSliderChange(val) {
    const stepIdx = parseInt(val, 10);
    if (!store.currentInference) return;

    const l1 = store.currentInference.levels.l1_letter;
    const steps = l1.steps;
    if (stepIdx >= steps.length) return;

    const curr = steps[stepIdx];

    // Update tracker highlight
    const pills = document.querySelectorAll('.char-pill');
    pills.forEach((p, idx) => {
        if (idx === stepIdx) {
            p.classList.add('active');
            p.setAttribute('aria-current', 'step');
        } else {
            p.classList.remove('active');
            p.removeAttribute('aria-current');
        }
    });

    document.getElementById('slider-step-label').textContent = 
        `Step ${stepIdx + 1} / ${steps.length} (${JSON.stringify(curr.char)})`;

    // 1. Level 1 Update
    document.getElementById('l1-obs').textContent = JSON.stringify(curr.char);
    document.getElementById('l1-entropy').textContent = `${curr.entropy.toFixed(3)} nats`;

    const topPredContainer = document.getElementById('l1-top-predictions');
    topPredContainer.innerHTML = '';
    curr.top_predictions.forEach(p => {
        const row = document.createElement('div');
        row.className = 'bar-row';
        row.innerHTML = `
            <div class="bar-label">${p.symbol}</div>
            <div class="bar-track"><div class="bar-fill" style="width: ${(p.prob * 100).toFixed(1)}%;"></div></div>
            <div class="bar-val">${(p.prob * 100).toFixed(1)}%</div>
        `;
        describeProbabilityBar(row, 'next symbol', p.symbol === ' ' ? 'space' : p.symbol, p.prob);
        topPredContainer.appendChild(row);
    });

    // Map character position to active word & word-internal char index
    const wordsData = store.currentInference.levels.l3_word.words;
    let runningLen = 0;
    let activeWordIdx = 0;
    let wordInternalCharIdx = 0;

    for (let i = 0; i < wordsData.length; i++) {
        const wLen = wordsData[i].raw_word.length + 1;
        if (stepIdx < runningLen + wLen) {
            activeWordIdx = i;
            wordInternalCharIdx = stepIdx - runningLen;
            break;
        }
        runningLen += wLen;
    }
    const activeWord = wordsData[activeWordIdx] || wordsData[0];

    // 2. Level 2 Update
    document.getElementById('l2-word').textContent = activeWord.raw_word;
    document.getElementById('l2-decoded-phones').textContent = activeWord.decoded_phones.join(' ');
    document.getElementById('l2-gold-phones').textContent = activeWord.gold_phones.join(' ');

    const phoneStepBars = document.getElementById('l2-phone-step-bars');
    phoneStepBars.innerHTML = '';
    if (activeWord.phone_steps && activeWord.phone_steps.length > 0) {
        const pStep = activeWord.phone_steps[Math.min(wordInternalCharIdx, activeWord.phone_steps.length - 1)];
        pStep.top_phones.forEach(tp => {
            const row = document.createElement('div');
            row.className = 'bar-row';
            row.innerHTML = `
                <div class="bar-label">${tp.phone}</div>
                <div class="bar-track"><div class="bar-fill bar-fill-cyan" style="width: ${(tp.prob * 100).toFixed(1)}%;"></div></div>
                <div class="bar-val">${(tp.prob * 100).toFixed(1)}%</div>
            `;
            describeProbabilityBar(row, 'phoneme posterior', tp.phone, tp.prob);
            phoneStepBars.appendChild(row);
        });
    }

    // 3. Level 3 Update
    const vocabularyStatus = document.getElementById('l3-in-vocab');
    vocabularyStatus.textContent = activeWord.in_vocab ? 'In lexicon' : 'Out of lexicon';
    vocabularyStatus.className = 'stat-val ' + (activeWord.in_vocab ? 'tint-green' : 'tint-orange');
    document.getElementById('l3-word-idx').textContent = `Word ${activeWordIdx + 1} of ${wordsData.length}`;

    const trieContainer = document.getElementById('l3-trie-candidates');
    trieContainer.innerHTML = '';
    activeWord.candidate_trie_words.forEach(cw => {
        const chip = document.createElement('div');
        chip.className = 'word-chip' + (cw === activeWord.clean_word ? ' active' : '');
        chip.textContent = cw;
        trieContainer.appendChild(chip);
    });

    // 4. Level 4 Update. A tag the lexicon did not have is marked, so a
    // defaulted NOUN is never mistaken for an observed one.
    const uposEl = document.getElementById('l4-upos');
    uposEl.textContent = activeWord.upos +
        (activeWord.upos_source === 'fallback' ? ' (default)' :
         activeWord.upos_source === 'punctuation' ? ' (skipped)' : '');
    uposEl.className = 'stat-val ' + (activeWord.upos_source === 'lexicon' ? 'tint-orange' : 'tint-muted');
    document.getElementById('l4-top-case').textContent =
        activeWord.case_bearing ? activeWord.top_case : '— not case-bearing';

    const caseContainer = document.getElementById('l4-case-bars');
    caseContainer.innerHTML = '';
    for (const [caseName, prob] of Object.entries(activeWord.case_posterior)) {
        const row = document.createElement('div');
        row.className = 'bar-row';
        row.innerHTML = `
            <div class="bar-label">${caseName}</div>
            <div class="bar-track"><div class="bar-fill bar-fill-purple" style="width: ${(prob * 100).toFixed(1)}%;"></div></div>
            <div class="bar-val">${(prob * 100).toFixed(1)}%</div>
        `;
        describeProbabilityBar(row, 'case posterior', caseName, prob);
        caseContainer.appendChild(row);
    }

    // 5 & 6. Level 5 & 6 Update
    const l5 = store.currentInference.levels.l5_clause;
    document.getElementById('l5-clause-type').textContent = l5.predicted_type.toUpperCase();

    const clauseContainer = document.getElementById('l5-clause-bars');
    clauseContainer.innerHTML = '';
    l5.types.forEach((tName, tIdx) => {
        const prob = l5.posterior[tIdx];
        const row = document.createElement('div');
        row.className = 'bar-row';
        row.innerHTML = `
            <div class="bar-label">${tName.slice(0, 4)}</div>
            <div class="bar-track"><div class="bar-fill bar-fill-blue" style="width: ${(prob * 100).toFixed(1)}%;"></div></div>
            <div class="bar-val">${(prob * 100).toFixed(1)}%</div>
        `;
        describeProbabilityBar(row, 'clause posterior', tName, prob);
        clauseContainer.appendChild(row);
    });

    // The widened surface sequence, which the markup has always had a
    // slot for and the script never filled.
    const widenedEl = document.getElementById('l5-widened-obs');
    if (widenedEl) {
        widenedEl.textContent = l5.has_evidence
            ? l5.widened_symbols_seq.map(s => l5.symbol_labels[s]).join(' → ')
            : 'no case-bearing or verbal token — type prior only';
    }

    const l6 = store.currentInference.levels.l6_narrative;
    document.getElementById('l6-narrative-phase').textContent =
        `${l6.current_phase.toUpperCase()} (Tercile ${l6.phase_idx + 1}/3)`;

    // P(type | phase), likewise previously unrendered.
    const biasEl = document.getElementById('l6-phase-bias');
    if (biasEl) {
        biasEl.textContent = l5.types
            .map((t, i) => `${t.slice(0, 4)} ${(l6.phase_type_emission[i] * 100).toFixed(1)}%`)
            .join('  ');
    }

    // The top-down channel itself: interrogative mass under D'(alpha)
    // at the selected weight, per phase, against the phase-reversed
    // control. At alpha = 0 every phase and the control must read
    // identically — that is the identity anchor, visible rather than
    // asserted.
    const sweepEl = document.getElementById('l6-channel-sweep');
    if (sweepEl && l6.channel_sweep) {
        let nearest = l6.channel_sweep[0];
        for (const row of l6.channel_sweep) {
            if (Math.abs(row.alpha - l6.alpha) < Math.abs(nearest.alpha - l6.alpha)) nearest = row;
        }
        sweepEl.innerHTML = '';
        l6.phases.forEach(ph => {
            const real = nearest.phases[ph], ctrl = nearest.scrambled[ph];
            const row = document.createElement('div');
            row.className = 'bar-row';
            row.innerHTML = `
                <div class="bar-label">${ph.slice(0, 4)}</div>
                <div class="bar-track"><div class="bar-fill bar-fill-lilac" style="width: ${(real * 100 * 3).toFixed(1)}%;"></div></div>
                <div class="bar-val">${(real * 100).toFixed(2)}%</div>
                <div class="bar-val bar-val-muted">ctl ${(ctrl * 100).toFixed(2)}%</div>
            `;
            row.setAttribute('role', 'listitem');
            row.setAttribute(
                'aria-label',
                `${ph} phase: ${(real * 100).toFixed(2)} percent; `
                    + `phase-reversed control ${(ctrl * 100).toFixed(2)} percent`
            );
            row.querySelector('.bar-track')?.setAttribute('aria-hidden', 'true');
            sweepEl.appendChild(row);
        });
        const note = document.createElement('div');
        note.className = 'chart-note';
        note.textContent = nearest.alpha === 0
            ? 'alpha = 0: identity anchor — every phase and the control return the unmodified clause prior.'
            : `alpha = ${nearest.alpha.toFixed(2)}: "ctl" is the phase-reversed control, which fixes the middle phase by construction.`;
        sweepEl.appendChild(note);
    }
}

export { replyLength, runInference, setupTimeline, onSliderChange };
