#!/usr/bin/env python3
"""Refresh dependencies and rebuild Docker image."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    run(["docker", "compose", "build", "app"])


if __name__ == "__main__":
    main()
