# Speech Boundary: Where DSP Ends and pymdp Begins

## The Boundary

The shipped [speech boundary module](../../src/ailm/speech_boundary.py) validates
phone symbols. The diagram below describes the separation an external audio
integration would need; it is not an implemented audio pipeline:

```
Audio signal
    ↓ [external audio-to-ARPAbet adapter — not implemented here]
space-separated ARPAbet
    ↓ [AILM boundary parser]
Level 2 phone symbols
    ↓ [independent AILM model use, if orchestrated by a caller]
Level 2 phone symbols
    ↓ [AILM boundary serializer]
space-separated ARPAbet
    ↓ [external symbol-to-audio adapter — not implemented here]
Audio signal (synthesized)
```

For a proposed phoneme speech adapter, pymdp starts at the phoneme level
after external acoustic conversion; a caller would have to orchestrate that
model use. Stage 7 itself remains a symbol parser and serializer.

**This boundary accepts already symbolized input.** The AILM agents do not
process audio, spectrograms, mel-filterbanks or waveforms. Raw-text connected
inference starts from spelling; it is a separate interface. Stage 7 itself
parses and serializes the boundary vocabulary and rejects invalid symbols. It
does not call Whisper, a TTS engine, the Level 2 agent, or a full hierarchy.

## Why This Boundary

The implemented model has finite symbol alphabets and discrete state factors.
Keeping acoustic conversion outside that model makes the input contract testable
without claiming an acoustic likelihood or a speech-quality result. A joint
acoustic-language model was not adopted; it would need its own representations,
training data, observation contract and evaluation.

An external speech recognizer's transcription is not automatically a valid
ARPAbet sequence. An adapter must explicitly produce the boundary vocabulary and
meet its refusal rules. The browser demonstrator's speech controls supply text
and browser speech output; they do not establish this phone-level audio bridge
or run the connected runtime. See the [dashboard scope](../../src/ailm/dashboard/README.md).

## Implemented contract (Stage 7)

Stage 7 exposes two fail-closed transformations:

```python
from_boundary("K AE1 T")  # -> ["K", "AE", "T"]
to_boundary(["K", "AE", "T"])  # -> "K AE T"
```

The vocabulary is the Level 2 stress-free ARPAbet inventory by construction.
Input may carry ordinary stress suffixes, which are stripped; unknown tokens and
empty sequences raise `BoundaryError`. Round-trip identity is checked over the
vendored pronunciations. A real audio wrapper and any orchestration into or out
of a phoneme model remain external work. The [connected runtime](connected_runtime.md)
infers position-aligned phones from spelling; it does not accept a supplied phone
sequence as a substitute for that spelling stream.

## What DSP Handles

- Raw audio capture and playback
- Noise reduction and normalization
- Speech activity detection
- Whisper inference (GPU or CPU)
- TTS synthesis (e.g., Coqui, pyttsx3, or ElevenLabs API)
- Prosody and duration modeling (future: could be modeled in pymdp, but out of scope)

## What AILM can handle on the model side

- A caller can pass parsed phone symbols to the independently built Level 2
  phone model.
- A caller can serialize valid phone symbols produced elsewhere.
- The current boundary does not promise phoneme generation, a connected
  phoneme→word→clause path, or a document-to-speech generation cascade.

## Stage 7 Dependency

The interface contract is implemented because the Level 2 inventory exists. An
end-to-end speech system would additionally require chosen STT and TTS adapters,
quality criteria for those external models, and explicit orchestration between
the boundary and whichever AILM model path is intended.
