# Third-party data notices

This file accompanies the component summary in [`LICENSE`](LICENSE). The
derivation paths and exact committed checksums are in
[`docs/provenance/sources.md`](docs/provenance/sources.md).

## Stage 1 evaluation protocol fixture

`ailm.evaluation.STAGE1_EVALUATION` is an AILM-curated, normalized selection of
an AILM-authored rhyming sentence and short traditional or public-domain English
phrases. The AILM selection and normalization are distributed under CC BY-SA
4.0; the incorporated historical phrases remain public domain in the United
States. The complete protocol identity, repetition disclosure, and source links
are recorded in `docs/provenance/sources.md`. This fixture is not derived from
the Brown Corpus.

## Brown Corpus via NLTK

- Authors: W. N. Francis and H. Kucera.
- Upstream package: `brown` in the NLTK data distribution.
- Exact source revision: `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a`;
  the immutable `packages/corpora/brown.zip` URL, digest, and byte extent are
  recorded in `docs/provenance/sources.md` and embedded in every derived artifact.
- Upstream terms: “May be used for non-commercial purposes.”
- Source metadata:
  <https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md>
- AILM transformation: normalized corpus text, counts, held-out streams, online
  drift material, word statistics, and clause/word statistics stored under
  `vendor/brown_*.json`.

The prior AILM notice called Brown public domain. That was incorrect; the
official NLTK package metadata explicitly places it in the non-commercial-use
group.

## Universal Dependencies English EWT

- Upstream project: Universal Dependencies English Web Treebank.
- License: CC BY-SA 4.0.
- Source: <https://universaldependencies.org/treebanks/en_ewt/>
- AILM transformation: deterministic case, clause, and narrative count artifacts
  under `vendor/ud_ewt_*.json`; the retained fields and exclusions are documented
  in `docs/provenance/sources.md`.

## Universal Dependencies English GUM

- Upstream project: Georgetown University Multilayer corpus, UD English-GUM;
  credits and source-text attributions are maintained by the upstream project.
- Exact source revision: `b58e74bc22d17220c9198864c253a50a897bf27f`.
- License: CC BY-NC-SA 4.0; the [pinned upstream notice](https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/b58e74bc22d17220c9198864c253a50a897bf27f/LICENSE.txt)
  explains the underlying text licenses and requires source and annotator credit.
- AILM includes the byte-identical `vendor/en_gum-ud-test.conllu` evaluation input.
  Source metadata and attributions remain in that file. The connected evaluation
  reads only its raw text comments; annotation labels are withheld from inference.
- The evaluation protocol and input manifest live under `data/connected_gum/`.
  This evaluation input ships with the source distribution, not the runtime wheel.

## Universal Dependencies Czech-PDTC

- Upstream project: UD Czech-PDTC; contributor credits are maintained on the
  upstream treebank page.
- Exact source revision: `6d206ec7d337a7f76f34ddfc82893389cabbd76d`;
  the fitting `train-ft` shard and held-out `dev` shard identities are embedded
  separately in the derived artifact.
- License: CC BY-NC-SA 4.0.
- Source: <https://universaldependencies.org/treebanks/cs_pdtc/>
- AILM transformation: the train-ft and dev shards were converted into
  `vendor/ud_czech_case_counts.json`, retaining encoded UPOS and morphological
  case counts while excluding token text and the untouched test shard.

## CMU Pronouncing Dictionary

- Upstream project: CMUdict.
- Source: <https://github.com/cmusphinx/cmudict>
- AILM transformation: pronunciation entries were normalized and aggregated into
  `vendor/cmudict_counts.json`.

The upstream BSD-2-Clause notice is retained below:

```text
Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   The contents of this file are deemed to be source code.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the
   distribution.
This work was supported in part by funding from the Defense Advanced
Research Projects Agency, the Office of Naval Research and the National
Science Foundation of the United States of America, and by member
companies of the Carnegie Mellon Sphinx Speech Consortium. We acknowledge
the contributions of many volunteers to the expansion and improvement of
this dictionary.
THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
