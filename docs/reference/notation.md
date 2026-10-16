# Notation Table

Consolidated symbol-to-meaning reference for the AILM manuscript, decision
records, and code. Each row names a symbol, its meaning, and the code path(s)
where it is defined or used. Symbols are grouped by function.

## Generative model matrices

| Symbol | Meaning | Defined in | Used in |
|--------|---------|------------|---------|
| `A` | Emission matrix: `P(observation \| state)` | `src/ailm/levels/*.py` | Every level builder |
| `B` | Transition matrix: `P(state_t \| state_{t-1}, action)` | `src/ailm/levels/*.py` | Every level builder |
| `C` | Log preferences over outcomes | `src/ailm/levels/letter.py`, `src/ailm/generation.py` | L1g generation; the document-position→clause intervention instead changes L5’s prior |
| `D` | Initial state distribution (prior) | `src/ailm/levels/*.py` | `build_g2p_start`, `build_phone_unigram` |
| `E` | Policy prior (habit) | `src/ailm/generation.py` | Generation — Brown bigram as policy prior |

## Level-specific matrices

| Symbol | Meaning | Level | Code |
|--------|---------|-------|------|
| `B_letter` | Brown bigram letter transition | L1 — Letters | `src/ailm/levels/letter.py::build_letter_agent` |
| `B_phone` | CMUdict phone bigram transition | L2A — Phonemes | `src/ailm/levels/phoneme.py::build_phone_transition` |
| `A_g2p` | Position-aligned letter-to-phone (G2P) emission: `P(letter \| phone)` | L2B — G2P | `src/ailm/levels/phoneme.py::build_g2p_emission` |
| `B_g2p` | G2P phone bigram within a word | L2B — G2P | `src/ailm/levels/phoneme.py::build_g2p_transition` |
| `B_word` | Identity (word identity factor) | L3 — Words | `src/ailm/levels/word.py` |
| `B_pos` | `+1` position shift | L3 — Words | `src/ailm/levels/word.py` |
| `A_case` | `P(UPOS \| case role)` | L4 — Cases | `src/ailm/levels/case.py::build_case_emission` |
| `B_case` | Case-role transition | L4 — Cases | `src/ailm/levels/case.py::build_case_transition` |
| `B_type` | Widened-symbol transitions conditional on persistent sentence type | L5 — canonical clauses | `src/ailm/levels/widened.py::build_widened_agent` |
| Case-only type chains | Retained ablation over case roles alone | L5 — historical ablation | `src/ailm/levels/clause.py::build_type_models` |
| `B_phase` | Phase (tercile) transition | L6 — Narrative | `src/ailm/levels/narrative.py::build_phase_models` |

## Filtering and policy symbols (manuscript formalism)

One meaning per symbol — the manuscript's `tbl:notation` in
`docs/manuscript/02_background.md` is the authoritative copy. The predictive prior
is `ρ_t`, not `π`, precisely so that `π` can mean policy unambiguously.

| Symbol | Meaning | Defined in |
|--------|---------|------------|
| `π` | Policy (sequence of actions) | `eq:efe`, `eq:generative` |
| `ρ_t` | One-step predictive prior over states | `eq:filtering` |
| `q_t` | Filtered posterior after observing `o_t` | `eq:filtering` |
| `q̃` | Policy-conditioned predictive over `(o_τ, s_τ)` | `eq:efe` |
| `p̃` | Preference distribution; `C_i = log p̃(o = i)` | `eq:pomdp` |
| `x ⪰ t` | Declared (inclusive) threshold satisfaction | `eq:gate_satisfies` |
| `x ≻ t` | Strict satisfaction, the audit relation (`gate_passes` in `src/ailm/gates.py`) | `eq:gate_satisfies` |
| `verdict(t; P)` | Gate verdict indexed by evaluation population `P` | `eq:gate_verdict` |

`tests/test_notation_registry.py` enforces the one-meaning-per-symbol promise on
exactly the symbols the manuscript's `tbl:notation` registers, and checks in
particular that `π` never carries a time subscript — the clash the registry was
built to prevent.

### The one deliberate exception: `α`

`α` is the single symbol the manuscript reuses on purpose, which is why it is
**not** registered and not scanned. Four distinct uses, each fixed by its
subscript or by the sentence that introduces it:

| Use | Where |
|---|---|
| Channel mixing weight | the coupling, lexical, and punctuation-type-oracle → word-prior sweeps (`levels/coupling.py`, `levels/lexical_channel.py`, `levels/clause_channel.py`) |
| Significance level | the multiplicity accounting (`src/ailm/statistics.py`) |
| `α_t`, the forward-algorithm message | the verification section (`verification/oracle.py::forward_filter`) |
| `α_c`, AILM's NOM/ACC scalar-scaffold likelihood weight | `eq:case_free_energy` (`src/ailm/cases.py::AILM_SCALAR_CASE_WEIGHTS`) |

The reuse is stated in `docs/manuscript/02_background.md` above the registry table and
disclaimed in the table's own caption, so a reader is told before they can be
misled. Do not "fix" it by registering `α` — registering it would make the
registry's guarantee false rather than true.

## Verdict classes (gate taxonomy)

| Label | Meaning | Code |
|-------|---------|------|
| `VACUOUS` | The trivial baseline already clears the threshold | `src/ailm/gates.py::classify_gate` — the single classifier; `scripts/audit_gates.py` reaches it through `src/ailm/gates.py::evaluate_gate` and `src/ailm/figures/gates.py::_classify` is a wrapper over it |
| `UNREACHABLE` | A bound-typed comparator cannot clear the threshold, licensing an impossibility claim on that population | Same |
| `ABOVE_REFERENCE` | An exceedable reference or privileged oracle cannot clear the threshold; no impossibility claim follows | Same |
| `UNAUDITED` | The baseline fails, but no ceiling and comparator type are available | Same |
| `BRACKETED_BOUND` | The baseline fails and a bound-typed comparator clears the threshold | Same |
| `BRACKETED_REFERENCE` | The baseline fails and an exceedable reference or oracle clears the threshold | Same |

## Metrics

| Symbol | Meaning | Codesite |
|--------|---------|----------|
| `H(X \| Y)` | Conditional entropy (nats) | `scripts/audit_gates.py::_conditional_entropy` |
| `ppl` | Perplexity: `exp(H)` | `src/ailm/levels/letter.py::compute_perplexity`, `src/ailm/verification/oracle.py` |
| `TV` | Total variation distance to true word distribution | Canonical: `src/ailm/generation.py::exact_lexicon_total_variation`, `::exact_ancestral_lexicon_total_variation`; sampled diagnostic: `::lexicon_total_variation` |
| `McNemar p` | Two-sided exact binomial test on discordant pairs | Central helper `src/ailm/statistics.py::exact_two_sided_binomial_p`; validated benchmark wrapper `src/ailm/benchmarks/scoring.py::exact_mcnemar_p` |
| `EFE G(π)` | Expected free energy under policy π | `src/ailm/generation.py::expected_free_energy_terms` (both terms, per action); the closed-form policy posterior is `::efe_policy_posterior` |
| `F_AILM-scalar(q;c)` | AILM's project-owned NOM/ACC scalar approximation; not a source formula for all cases | `src/ailm/cases.py::case_conditioned_free_energy` (implemented but inert) |

## Corpus statistics

| Symbol | Meaning | Source |
|--------|---------|--------|
| `N_SYMBOLS = 30` | Letter alphabet (a–z) + space + period + apostrophe + hyphen | `src/ailm/levels/letter.py::ALPHABET` |
| `N_PHONES = 39` | Canonical ARPAbet size | `src/ailm/corpus/cmudict.py::CANONICAL_ARPABET` |
| `N_CASES = 8` | CEREBRUM case roles | `src/ailm/cases.py::CaseRole` |
| `N_TYPES = len(CLAUSE_TYPES)` | Retained sentence types: declarative and interrogative | `src/ailm/levels/clause.py` |
| `N_PHASES = 3` | Sentence-position terciles | `src/ailm/levels/narrative.py` |
| `N_VOCAB = 2000` | Vocabulary size (most frequent alphabetic Brown types) | `src/ailm/levels/word.py` |

## Special observations

| Observation | Meaning | Used in |
|-------------|---------|---------|
| `UPOS` | Universal Dependencies part-of-speech tag | L4 — Cases; the only field the case model observes |
| `deprel` | Dependency relation (withheld from model) | Evaluation oracle only; labels derived via `UD_DEPREL_TO_CASE` |
| `feats[Case]` | Morphological case feature (withheld from model) | Gold-morphology evaluation only |

## Connected runtime trace

`InferenceEvent` in `src/ailm/levels/connected_types.py` records the event clock
(`sentence`, `token`, `position`), pre-observation `prior` and `predictive`, the
available `observation`, and post-update `posterior`. These distributions have
level-specific state and observation axes; they are not one global posterior.
`Connections` names enabled/disabled/shuffled interventions. Missing bottom-up
evidence retains the receiver prior, followed by its scheduled temporal advance.
The [composition contract](../design/connected_runtime.md) defines the clocks
and raw versus structured observation availability.

## Manuscript tokens

Token = ledger label uppercased; no second mapping table.

| Form | Example | Resolution |
|------|---------|------------|
| `{{STAGE1_LETTER_PERPLEXITY}}` | `{{UPPERCASED_LABEL}}` | → `data/claims.json` via `src/ailm/manuscript_variables/` |
| `claim:phoneme_count_39` | `claim:<label>` | → `data/claims.json` via `src/ailm/claims.py::ClaimsLedger.scan_prose` |

## Code paths (key functions)

| Function | Purpose | File |
|----------|---------|------|
| `build_letter_agent` | Construct L1 letter bigram pymdp agent | `src/ailm/levels/letter.py` |
| `build_phoneme_agent` | Construct L2A phoneme sequence agent | `src/ailm/levels/phoneme.py` |
| `build_g2p_agent` | Construct L2B G2P agent | `src/ailm/levels/phoneme.py` |
| `build_word_prior` | Construct L3 word × position prior | `src/ailm/levels/word.py` |
| `build_case_*` | Construct L4 case-level matrices | `src/ailm/levels/case.py` |
| `build_type_models` | Construct retained case-only L5 ablation chains | `src/ailm/levels/clause.py` |
| `load_widened_clause_artifact`, `build_widened_agent`, `fit_markov_chain` | Load and construct canonical widened L5 chains/agent | `src/ailm/levels/widened.py` |
| `build_phase_models` | Construct L6 narrative-level phase models | `src/ailm/levels/narrative.py` |
| `generate_by_policy_selection` | Generate from the EFE policy posterior | `src/ailm/generation.py` |
| `build_agent_from_spec` | Dispatch a shipped GNN model to its registered builder | `src/ailm/spec_agent.py` |
| `load_ledger` | Load claims ledger | `src/ailm/claims.py` |
| `verify_decoder_equivalence` | Check bounded pymdp/NumPy decoded-position agreement | `scripts/record_stage2_claims.py` |
