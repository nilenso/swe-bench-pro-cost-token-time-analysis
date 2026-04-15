"""
Analysis package for SWE-Bench Pro trajectory data.

Layered architecture:
  models.py       — taxonomy definitions (single source of truth)
  classify.py     — thin wrapper around scripts/classify_intent.py
  orchestrate.py  — file collection, parallel processing, caching
  aggregate.py    — metrics computation from per-file results
  cli.py          — command-line entry point
"""
