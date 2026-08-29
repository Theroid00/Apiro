"""
apiro/run.py — packaging entry point.

`pip install -e .` registers `apiro = apiro.run:main` as a console script
(see pyproject.toml). This module exists to satisfy that entry point; the
actual CLI logic lives in scripts/investigate.py so both `python
scripts/investigate.py ...` (works straight from a clone, no install
required) and the installed `apiro` command stay in sync with a single
implementation.

This only works for an editable install (`pip install -e .`) run from
within the cloned repo, since `scripts/` isn't a packaged module — it adds
the repo's `scripts/` directory to sys.path at call time and imports
investigate.py from there. For anything else, run
`python scripts/investigate.py` or `uvicorn scripts.app:app` directly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    scripts_dir = ROOT / "scripts"
    if not scripts_dir.exists():
        print(
            "[-] Could not find scripts/investigate.py next to this installation.\n"
            "    The `apiro` command only works for an editable install run from\n"
            "    inside the cloned repository (`pip install -e .`).\n"
            "    Otherwise, run `python scripts/investigate.py` directly from the repo."
        )
        sys.exit(1)

    sys.path.insert(0, str(scripts_dir))
    import investigate  # scripts/investigate.py

    investigate.main()


if __name__ == "__main__":
    main()
