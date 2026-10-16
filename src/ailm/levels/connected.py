"""Causal multirate composition of the shipped live pymdp linguistic agents.

This is a modular conditional filter, not exact inference in a global joint
model. Incoming posterior messages are explicit interventions. Categorical
observations use pymdp's expected-log-likelihood convention; they are not
marginal likelihoods. The composition contract is in docs/design/connected_runtime.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ailm.levels.connected_builder import ConnectedModel
from ailm.levels.connected_types import (
    Channel,
    ConnectedInputError,
    Connections,
    DocumentResult,
    GeneratedWord,
    InferenceEvent,
    PolicyEvent,
    Sentence,
    TokenResult,
    parse_text,
)
from ailm.levels.letter import ALPHA_LIST, ALPHA_TO_IDX, N_SYMBOLS
from ailm.levels.word import END_INDEX

if TYPE_CHECKING:
    from pymdp.agent import Agent


def _agent_infer(agent: Agent, observation: jax.Array, prior: list[jax.Array]) -> list[jax.Array]:
    """Call pymdp itself; the module-level compiled wrapper caches its executable."""
    return agent.infer_states([observation], prior, preprocess_fn=lambda observations: observations)


def _agent_advance(agent: Agent, posterior: list[jax.Array]) -> list[jax.Array]:
    return agent.update_empirical_prior(jnp.zeros((1, len(posterior)), dtype=jnp.int32), posterior)


def _agent_policy(agent: Agent, state: jax.Array) -> tuple[jax.Array, jax.Array]:
    posterior, _ = agent.infer_policies([state])
    return posterior, agent.sample_action(posterior)


_compiled_infer = eqx.filter_jit(_agent_infer)
_compiled_advance = eqx.filter_jit(_agent_advance)
_compiled_policy = eqx.filter_jit(_agent_policy)


def _probability(values: NDArray[np.float64]) -> NDArray[np.float64]:
    total = float(values.sum())
    if not np.isfinite(values).all() or (values < 0).any() or total <= 0:
        raise ConnectedInputError("nonfinite, negative or zero probability mass")
    return values / total


def _start(agent: Agent) -> tuple[NDArray[np.float64], ...]:
    return tuple(_probability(np.asarray(p, dtype=np.float64).ravel()) for p in agent.D)


def _advance(
    agent: Agent, posterior: tuple[NDArray[np.float64], ...], *, compiled: bool
) -> tuple[NDArray[np.float64], ...]:
    qs = [jnp.asarray(q)[None, None, :] for q in posterior]
    advance = _compiled_advance if compiled else _agent_advance
    prior = advance(agent, qs)
    return tuple(_probability(np.asarray(p, dtype=np.float64).ravel()) for p in prior)


def _predict(agent: Agent, prior: tuple[NDArray[np.float64], ...]) -> NDArray[np.float64]:
    predictive = np.asarray(agent.A[0], dtype=np.float64)[0]
    for distribution in reversed(prior):
        predictive = np.tensordot(predictive, distribution, axes=(-1, 0))
    return _probability(predictive)


def _infer(
    agent: Agent,
    prior: tuple[NDArray[np.float64], ...],
    observation: NDArray[np.float64] | None,
    level: str,
    clock: tuple[int, int, int],
    events: list[InferenceEvent],
    *,
    compiled: bool,
) -> tuple[NDArray[np.float64], ...]:
    """Run live inference with categorical evidence; preserve the pre-update prediction."""
    if observation is None:
        observation = np.zeros(int(agent.A[0].shape[1]), dtype=np.float64)
        posterior = tuple(p.copy() for p in prior)
    else:
        observation = _probability(observation)
        infer = _compiled_infer if compiled else _agent_infer
        qs = infer(
            agent,
            jnp.asarray(observation)[None, :],
            [jnp.asarray(p)[None, :] for p in prior],
        )
        posterior = tuple(_probability(np.asarray(q, dtype=np.float64).ravel()) for q in qs)
    events.append(
        InferenceEvent(level, *clock, prior, observation, _predict(agent, prior), posterior)
    )
    return posterior


class ConnectedRuntime:
    """Reusable immutable model with document-local mutable inference state.

    Each call owns its entire state; reusing this object cannot carry a belief
    across documents. The fixed channel strength is an engineering setting,
    not an empirically selected optimum. ``compile_agent_calls`` selects cached
    compilation of the same live Agent methods; the eager mode exists for
    numerical execution controls and does not change the model configuration.
    """

    def __init__(
        self,
        model: ConnectedModel,
        connections: Connections | None = None,
        *,
        compile_agent_calls: bool = True,
    ) -> None:
        self.model = model
        self.connections = Connections() if connections is None else connections
        self.compile_agent_calls = compile_agent_calls

    def _message(self, channel: Channel, message: NDArray[np.float64]) -> NDArray[np.float64]:
        if channel in self.connections.shuffled:
            return message[::-1].copy()
        return message

    def _mix(
        self, channel: Channel, base: NDArray[np.float64], message: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if channel in self.connections.disabled or self.connections.strength == 0:
            return base.copy()
        alpha = self.connections.strength
        return _probability((1 - alpha) * base + alpha * self._message(channel, message))

    def _observation(
        self, channel: Channel, message: NDArray[np.float64]
    ) -> NDArray[np.float64] | None:
        if channel in self.connections.disabled or self.connections.strength == 0:
            return None
        return _probability(self._message(channel, message))

    def _clause_start(self, phase_prior: NDArray[np.float64]) -> NDArray[np.float64]:
        model = self.model
        joint = _start(model.clause)[0].reshape(2, -1)
        type_prior = joint.sum(axis=1)
        incoming = _predict(model.phase, (phase_prior,))
        types = self._mix(Channel.PHASE_CLAUSE, type_prior, incoming)
        return (joint / type_prior[:, None] * types[:, None]).ravel()

    def _case_prior(
        self, case_prior: NDArray[np.float64], clause_prior: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        message = clause_prior.reshape(2, -1).sum(axis=0)[: len(case_prior)]
        return self._mix(Channel.CLAUSE_CASE, case_prior, _probability(message))

    def _word_prior(self, case: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._mix(
            Channel.CASE_WORD, _start(self.model.word)[0], self.model.word_given_case @ case
        )

    def _letter_message(self, word_prior: tuple[NDArray[np.float64], ...]) -> NDArray[np.float64]:
        lexical = _predict(self.model.word, word_prior)
        message = np.zeros(N_SYMBOLS, dtype=np.float64)
        for i in range(END_INDEX):
            message[ALPHA_TO_IDX[chr(97 + i)]] = lexical[i]
        message[ALPHA_TO_IDX[" "]] = lexical[END_INDEX]
        return message

    def infer_text(self, text: str) -> DocumentResult:
        """Infer a raw document under the explicit ASCII lexical normalization."""
        return self.infer_document(parse_text(text))

    def infer_document(self, sentences: tuple[Sentence, ...]) -> DocumentResult:
        """Filter a document without any evaluation labels or future-token features."""
        if not sentences:
            raise ConnectedInputError("a document must contain a sentence")
        model = self.model
        for sentence in sentences:
            for token in sentence.tokens:
                if token.upos is not None and token.upos not in model.upos:
                    raise ConnectedInputError(f"unknown UPOS {token.upos!r}")
        events: list[InferenceEvent] = []
        results: list[TokenResult] = []
        types: list[NDArray[np.float64]] = []
        phase_prior = _start(model.phase)
        phase_posterior = phase_prior
        letter_prior = _start(model.letter)
        for si, sentence in enumerate(sentences):
            clause_prior: tuple[NDArray[np.float64], ...] = (self._clause_start(phase_prior[0]),)
            case_prior = _start(model.case)
            for ti, token in enumerate(sentence.tokens):
                clock = (si, ti, -1)
                incoming_case = (self._case_prior(case_prior[0], clause_prior[0]),)
                upos = model.unknown_upos.copy()
                case_posterior: tuple[NDArray[np.float64], ...] = incoming_case
                if token.upos is not None:
                    upos = np.eye(len(model.upos))[model.upos.index(token.upos)]
                    case_posterior = _infer(
                        model.case,
                        incoming_case,
                        upos,
                        "case",
                        clock,
                        events,
                        compiled=self.compile_agent_calls,
                    )
                word_prior: tuple[NDArray[np.float64], ...] = (
                    self._word_prior(case_posterior[0]),
                    _start(model.word)[1],
                )
                initial_word = word_prior[0].copy()
                word_index = model.lexicon.word_to_index.get(token.form)
                probability = 0.0 if word_index is None else float(initial_word[word_index])
                phone_prior = _start(model.phone)
                phones: list[NDArray[np.float64]] = []
                supported = True
                for position, char in enumerate(token.form):
                    clock = (si, ti, position)
                    if supported:
                        letter_prior = (
                            self._mix(
                                Channel.WORD_LETTER,
                                letter_prior[0],
                                self._letter_message(word_prior),
                            ),
                        )
                        lexical_phone = (
                            model.phone_given_word_position[
                                min(position, model.lexicon.n_positions - 1)
                            ]
                            @ word_prior[0]
                        )
                        phone_prior = (
                            self._mix(Channel.WORD_PHONE, phone_prior[0], lexical_phone),
                        )
                    observation = np.eye(N_SYMBOLS)[ALPHA_TO_IDX[char]]
                    letter_q = _infer(
                        model.letter,
                        letter_prior,
                        observation,
                        "letter",
                        clock,
                        events,
                        compiled=self.compile_agent_calls,
                    )
                    letter_prior = _advance(
                        model.letter, letter_q, compiled=self.compile_agent_calls
                    )
                    phone_q = _infer(
                        model.phone,
                        phone_prior,
                        observation,
                        "phone",
                        clock,
                        events,
                        compiled=self.compile_agent_calls,
                    )
                    phone_prior = _advance(model.phone, phone_q, compiled=self.compile_agent_calls)
                    phones.append(phone_q[0])
                    letter_index = ord(char) - 97
                    if supported and _predict(model.word, word_prior)[letter_index] > 1e-12:
                        word_q = _infer(
                            model.word,
                            word_prior,
                            np.eye(END_INDEX + 1)[letter_index],
                            "word",
                            clock,
                            events,
                            compiled=self.compile_agent_calls,
                        )
                        word_prior = _advance(model.word, word_q, compiled=self.compile_agent_calls)
                    else:
                        supported = False
                clock = (si, ti, len(token.form))
                if supported and _predict(model.word, word_prior)[END_INDEX] > 1e-12:
                    word_q = _infer(
                        model.word,
                        word_prior,
                        np.eye(END_INDEX + 1)[END_INDEX],
                        "word",
                        clock,
                        events,
                        compiled=self.compile_agent_calls,
                    )
                else:
                    supported = False
                if token.upos is None:
                    if supported:
                        upos = _probability(model.upos_given_word @ word_q[0])
                    case_posterior = _infer(
                        model.case,
                        incoming_case,
                        upos,
                        "case",
                        clock,
                        events,
                        compiled=self.compile_agent_calls,
                    )
                case_prior = _advance(model.case, case_posterior, compiled=self.compile_agent_calls)
                widened = np.zeros(int(model.clause.A[0].shape[1]), dtype=np.float64)
                verb = float(upos[model.upos.index("VERB")]) if "VERB" in model.upos else 0.0
                aux = float(upos[model.upos.index("AUX")]) if "AUX" in model.upos else 0.0
                widened[:-2] = case_posterior[0] * (1 - verb - aux)
                widened[-2:] = verb, aux
                clause_q = _infer(
                    model.clause,
                    clause_prior,
                    self._observation(Channel.CASE_CLAUSE, widened),
                    "clause",
                    clock,
                    events,
                    compiled=self.compile_agent_calls,
                )
                clause_prior = _advance(model.clause, clause_q, compiled=self.compile_agent_calls)
                results.append(
                    TokenResult(
                        token.form,
                        initial_word,
                        probability,
                        word_index is None,
                        tuple(phones),
                        case_posterior[0],
                        upos,
                    )
                )
                space_q = _infer(
                    model.letter,
                    letter_prior,
                    np.eye(N_SYMBOLS)[ALPHA_TO_IDX[" "]],
                    "letter",
                    clock,
                    events,
                    compiled=self.compile_agent_calls,
                )
                letter_prior = _advance(model.letter, space_q, compiled=self.compile_agent_calls)
            sentence_type = _probability(clause_q[0].reshape(2, -1).sum(axis=1))
            types.append(sentence_type)
            phase_posterior = _infer(
                model.phase,
                phase_prior,
                self._observation(Channel.CLAUSE_PHASE, sentence_type),
                "phase",
                (si, -1, -1),
                events,
                compiled=self.compile_agent_calls,
            )
            phase_prior = _advance(model.phase, phase_posterior, compiled=self.compile_agent_calls)
        next_case = self._case_prior(_start(model.case)[0], self._clause_start(phase_prior[0]))
        return DocumentResult(
            tuple(results),
            tuple(types),
            phase_posterior[0],
            self._word_prior(next_case),
            tuple(events),
        )

    def generate_next_word(
        self, context: DocumentResult, *, max_letters: int = 32
    ) -> GeneratedWord:
        """Select a next-sentence word with live policy inference and letter feedback.

        The context's next-word prior carries the completed document's inferred
        phase into a fresh sentence. This is bounded conditional generation,
        not a claim of autonomous document planning.
        """
        if isinstance(max_letters, bool) or max_letters < 1:
            raise ConnectedInputError("max_letters must be positive")
        model = self.model
        prior = _probability(np.asarray(context.next_word_prior, dtype=np.float64))
        if prior.shape != _start(model.word)[0].shape:
            raise ConnectedInputError("context word vocabulary does not match model")
        word_prior: tuple[NDArray[np.float64], ...] = (prior, _start(model.word)[1])
        current = ALPHA_TO_IDX[" "]
        prefix = ""
        policies: list[PolicyEvent] = []
        events: list[InferenceEvent] = []
        for position in range(max_letters + 1):
            mass = self._letter_message(word_prior)
            preferences = np.log(np.maximum(mass, 1e-16))
            habit = _probability(np.asarray(model.letter.B[0], dtype=np.float64)[0, :, current, 0])
            installed = eqx.tree_at(
                lambda agent: (agent.C, agent.E),
                model.generator,
                ([jnp.asarray(preferences)[None, :]], jnp.asarray(habit)[None, :]),
            )
            policy = _compiled_policy if self.compile_agent_calls else _agent_policy
            q_pi, selected_action = policy(
                installed, jnp.asarray(np.eye(N_SYMBOLS)[current])[None, None, :]
            )
            action = int(np.asarray(selected_action).ravel()[0])
            policies.append(
                PolicyEvent(prefix, preferences, habit, np.asarray(q_pi).ravel(), action)
            )
            char = ALPHA_LIST[action]
            if char == " ":
                return GeneratedWord(prefix, True, prior, tuple(policies))
            if position == max_letters:
                break
            if not char.isalpha():
                raise ConnectedInputError("policy selected a nonlexical symbol")
            word_q = _infer(
                model.word,
                word_prior,
                np.eye(END_INDEX + 1)[ord(char) - 97],
                "word",
                (0, 0, position),
                events,
                compiled=self.compile_agent_calls,
            )
            word_prior = _advance(model.word, word_q, compiled=self.compile_agent_calls)
            prefix += char
            current = action
        return GeneratedWord(prefix, False, prior, tuple(policies))
