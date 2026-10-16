# /// script
# requires-python = ">=3.11"
# ///
# Run: .venv/bin/python scripts/run_connected_ailm.py --text "the man."
"""Run the connected model on raw text or a strict lexical-token JSON document."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ailm.levels.connected import ConnectedRuntime  # noqa: E402
from ailm.levels.connected_builder import load_connected_model  # noqa: E402
from ailm.levels.connected_types import Channel, Connections, Sentence, Token  # noqa: E402


def main() -> int:
    """Parse observations and print strict JSON; errors exit nonzero."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--structured", type=Path, help="JSON array of sentences of {form, upos?}")
    parser.add_argument("--generate", action="store_true", help="select a next-sentence word")
    parser.add_argument("--trace", action="store_true", help="include full inference-event arrays")
    parser.add_argument("--disable", action="append", choices=list(Channel), default=[])
    parser.add_argument("--shuffle", action="append", choices=list(Channel), default=[])
    args = parser.parse_args()
    try:
        document = None
        if args.structured is not None:
            payload = json.loads(args.structured.read_text(encoding="utf-8"))
            document = tuple(
                Sentence(tuple(Token(**token) for token in sentence)) for sentence in payload
            )
        model = load_connected_model()
        runtime = ConnectedRuntime(
            model,
            Connections(
                disabled=frozenset(Channel(value) for value in args.disable),
                shuffled=frozenset(Channel(value) for value in args.shuffle),
            ),
        )
        result = (
            runtime.infer_text(args.text) if document is None else runtime.infer_document(document)
        )
        output = {
            "schema": "ailm-connected-runtime-v1",
            "scope": "causal conditional filtering; not a global Bayesian posterior",
            "normalization": "ASCII letter runs; punctuation marks boundaries but has no type label",
            "word_log_loss": result.word_log_loss if math.isfinite(result.word_log_loss) else None,
            "word_loss_status": "infinite_oov" if any(t.oov for t in result.tokens) else "finite",
            "letter_log_loss": result.letter_log_loss,
            "tokens": [
                {
                    "form": t.form,
                    "oov": t.oov,
                    "word_probability": t.word_probability,
                    "case_posterior": t.case_posterior.tolist(),
                    "phone_posteriors": [p.tolist() for p in t.phone_posteriors],
                }
                for t in result.tokens
            ],
            "sentence_types": [p.tolist() for p in result.sentence_types],
            "phase_posterior": result.phase_posterior.tolist(),
            "event_count": len(result.events),
        }
        if args.generate:
            output["generation"] = dataclasses.asdict(runtime.generate_next_word(result))
        if args.trace:
            output["events"] = [dataclasses.asdict(event) for event in result.events]
        print(json.dumps(output, allow_nan=False, default=lambda value: np.asarray(value).tolist()))
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, f"connected model input/runtime error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
