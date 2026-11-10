"""Works before editable installation, regardless of the current directory."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))
from ncrf.cli import main

if __name__=="__main__":
    main()
