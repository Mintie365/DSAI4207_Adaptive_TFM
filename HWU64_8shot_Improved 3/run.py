from __future__ import annotations

import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        "This project requires Python 3.10+ (recommended: 3.12). "
        "Your current Python is " + sys.version.split()[0] + "."
    )

from src.pipeline import main

if __name__ == "__main__":
    main()
