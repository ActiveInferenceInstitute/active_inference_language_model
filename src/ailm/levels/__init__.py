"""AILM hierarchical level modules.

Each level builds a pymdp generative model at a different linguistic granularity:
  - letter:  30-symbol bigram model (Brown corpus)
  - phoneme: ARPAbet transitions from CMUdict (Stage 2)
  - word:    trie-constrained word paths (Stage 3)
  - case:    8-case assignment (Stage 4)
  - clause:  temporal policy over case-labeled words (Stage 5)
  - narrative: clause sequence priors (Stage 6)
"""
