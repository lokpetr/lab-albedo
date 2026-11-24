import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent/'scripts'))
from process import main

if __name__ == '__main__':
    main()
