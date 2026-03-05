"""
Configure sys.path so tests can import from src.quantum.
"""
import sys
import os

# Add the repo root to sys.path so 'from src.quantum import ...' works.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
