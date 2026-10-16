"""Corpus loaders for AILM.

cmudict: CMU Pronouncing Dictionary — letter-to-phoneme bridge.
    claim:cmudict_pronunciation_entry_count entries and
    claim:phoneme_count_39 canonical ARPAbet phonemes.
    CRITICAL: Strip '#' comments before parsing or inventory is corrupted.

ud: Universal Dependencies English EWT — case oracle for Level 4.
    Gold Nom/Acc/Gen; DAT/INS/LOC/ABL/VOC derived via documented mapping.
    The derived mapping is a RESEARCH CLAIM — review before treating accuracy as ground truth.

Corpora are vendored to vendor/ with checksums in docs/provenance/sources.md.
They are excluded from git (see .gitignore).
"""
