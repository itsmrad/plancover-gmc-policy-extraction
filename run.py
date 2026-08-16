#!/usr/bin/env python3
"""Zero-install entry point.

The package uses a ``src/`` layout, so ``python -m gmc_extract`` only works once the project
has been installed. This shim puts ``src`` on the path first, so a reviewer can run the
pipeline immediately after ``pip install -r requirements.txt`` without an install step (and
without needing a pip new enough for PEP 660 editable installs).

    python run.py run
    python run.py run --input "data/input/GHI Policy.pdf" --output /tmp/out
    python run.py fields
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from gmc_extract.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
