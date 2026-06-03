"""
Unified training entry point.

Usage:
    python train.py                          # DQN training (default)
    python train.py --method baselines       # Baselines only
    python train.py --method dqn --data 6x6x2 --episodes 500
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    # Parse --method from command line, forward everything else
    method = "dqn"
    filtered = [sys.argv[0]]

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--method" and i + 1 < len(sys.argv):
            method = sys.argv[i + 1]
            i += 2
        else:
            filtered.append(sys.argv[i])
            i += 1

    sys.argv = filtered

    if method == "baselines":
        from train_baselines import main as run
    else:
        from train_dqn import main as run

    run()


if __name__ == "__main__":
    main()
