"""Make `import fire_mapper` work no matter where pytest is started from (no sourcing needed)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
