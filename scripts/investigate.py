#!/usr/bin/env python3
"""
scripts/investigate.py
======================
Backwards-compatible wrapper. The canonical CLI is now in `apiro.cli`.

Usage:
  python scripts/investigate.py --findings "49yo female, dyspnea..."
  python -m apiro.cli --findings "..."
"""
from apiro.cli import main, build_components

if __name__ == "__main__":
    main()
