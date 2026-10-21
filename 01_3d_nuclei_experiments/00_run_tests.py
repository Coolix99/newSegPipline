"""Run automated regression tests without arguments."""
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parent
raise SystemExit(subprocess.call([sys.executable,'-m','pytest','-q',str(root/'tests')],cwd=root))
