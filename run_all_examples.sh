#!/bin/bash

rm -rf ./results
mkdir results
cd results || exit

cases=("case1" "case2" "case3" "case4")

for case in "${cases[@]}"; do
  nohup python -u "../examples/validation/${case}.py" > "${case}.out" 2> "${case}.err" &
done

cases=("laminar_flame" "optimization")

for case in "${cases[@]}"; do
  nohup python -u "../examples/${case}.py" > "${case}.out" 2> "${case}.err" &
done
