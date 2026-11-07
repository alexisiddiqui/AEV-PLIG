"""Shared pytest setup: point AEVPLIG_REPO at the copied clone modules and put src/ on sys.path."""
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["AEVPLIG_REPO"] = os.path.join(TESTS_DIR, "fake_repo")
sys.path.insert(0, os.path.join(os.path.dirname(TESTS_DIR), "src"))
