"""Launch/status entrypoint for the paired SSAI learning suite."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.ssai_suite import main

if __name__ == '__main__':
    main()
