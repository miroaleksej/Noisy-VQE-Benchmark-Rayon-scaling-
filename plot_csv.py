#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) < 4:
        print("Usage: plot_csv.py <csv> <xcol> <ycol> [title]")
        return 1

    csv_path = Path(sys.argv[1])
    xcol = sys.argv[2]
    ycol = sys.argv[3]
    title = sys.argv[4] if len(sys.argv) > 4 else ycol

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1

    x = []
    y = []

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        if xcol not in reader.fieldnames or ycol not in reader.fieldnames:
            print(f"ERROR: columns not found. Available: {reader.fieldnames}")
            return 1
        for row in reader:
            x.append(float(row[xcol]))
            y.append(float(row[ycol]))

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    out = csv_path.with_name(f"{csv_path.stem}_{ycol}.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
