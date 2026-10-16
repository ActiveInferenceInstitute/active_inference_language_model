"""Generative model utilities for AILM.

smoke: pymdp API verification (A_dependencies, update_empirical_prior, etc.)
       These run REAL pymdp with no mocks. Stage 0 gate depends on them passing.

Stage 1+: letter model, phoneme model, word trie, clause/case layer, narrative.
Each stage is gated by its own falsifiable test before the next stage starts.
"""
