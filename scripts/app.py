#!/usr/bin/env python3
"""
scripts/app.py
==============
Backwards-compatible wrapper. The canonical app is now in `apiro.web.app`.

Run with:
  uvicorn apiro.web:app --host 127.0.0.1 --port 8000
  python -m apiro.web
"""
import uvicorn
from apiro.web.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
