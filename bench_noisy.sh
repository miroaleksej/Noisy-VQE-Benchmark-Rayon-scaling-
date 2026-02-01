#!/usr/bin/env bash
set -e

MODE=noisy
TRAJ=40
SHOTS=200
P=0.01
SEED=bench-seed

echo "Benchmark noisy VQE"
echo "traj=$TRAJ shots=$SHOTS p=$P"
echo

for T in 1 2 4 8; do
  echo "threads = $T"
  /usr/bin/time -f "  real %e s" \
    cargo run -p emulator --release -- \
      --mode $MODE \
      --threads $T \
      --trajectories $TRAJ \
      --shots $SHOTS \
      --p $P \
      --seed $SEED \
      > /dev/null
  echo
 done
