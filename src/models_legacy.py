"""
Compatibility shim — redirects all imports to the canonical ModelZoo.

Background:
  The original monolithic `models_legacy.py` was archived on 2026-03-16
  (see archive/src/models_legacy.py).  20+ scripts still import from this
  path.  This thin redirect keeps them working while preserving a single
  source of truth at `src.models.zoo.ModelZoo`.

Migration plan:
  Replace every `from src.models_legacy import ModelZoo`
  with    `from src.models import ModelZoo`
  then delete this file.

Created: 2026-03-17 (AI-PI Phase II — stack cleanup)
"""

from src.models.zoo import ModelZoo, SimpleMambaBlock  # noqa: F401

__all__ = ["ModelZoo", "SimpleMambaBlock"]
