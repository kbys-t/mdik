#!/bin/bash

tasks=(
  "regulation"
  "tracking"
)

names=(
  "talos"
  "tiago"
  "ur5_limited"
)

n_resume=0

for task in "${tasks[@]}" ; do
  for name in "${names[@]}" ; do
    python test.py --task $task --name $name --n_resume $n_resume
  done
done

python eval.py
