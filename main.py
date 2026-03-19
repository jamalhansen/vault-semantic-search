import sys
from pathlib import Path

# Add src to sys.path to allow importing from the internal package
sys.path.append(str(Path(__file__).parent / "src"))

from vsearch.logic import app

if __name__ == "__main__":
    app()
