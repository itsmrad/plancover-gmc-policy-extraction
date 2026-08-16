import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

SAMPLE_DIR = os.path.join(REPO_ROOT, "data", "input")


@pytest.fixture(scope="session")
def sample_dir():
    if not os.path.isdir(SAMPLE_DIR):
        pytest.skip("sample PDFs not present")
    return SAMPLE_DIR


@pytest.fixture(scope="session")
def documents(sample_dir):
    """Load every sample PDF once; loading is the slow part of the suite."""
    from gmc_extract.ingestion import discover_pdfs, load_document

    return {os.path.basename(path): load_document(path)
            for path in discover_pdfs(sample_dir)}
