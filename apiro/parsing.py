"""
apiro.parsing — backwards-compatibility alias.
Canonical implementation has moved to `apiro.eval.parsing`.
"""
from apiro.eval.parsing import *
from apiro.eval import parsing as _parsing

__all__ = _parsing.__all__
