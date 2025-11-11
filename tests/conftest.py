"""Pytest configuration file to set up import paths.
This ensures that the src/ directory is on sys.path when running tests.
This allows tests to import kameris_repo modules correctly.
For example, the kmer.py module provides functions to extract k-mer frequency vectors from DNA sequences"""

import os
import sys

# Ensure project's src/ is on sys.path for imports during tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
