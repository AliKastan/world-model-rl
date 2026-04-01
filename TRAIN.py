"""Train a PPO agent to solve Sokoban puzzles with curriculum learning.

Usage:
    python TRAIN.py                    # Default: 50k episodes
    python TRAIN.py --episodes 10000   # Quick training run
    python TRAIN.py --resume ckpt.pt   # Resume from checkpoint
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from training.train_sokoban import main

if __name__ == "__main__":
    main()
