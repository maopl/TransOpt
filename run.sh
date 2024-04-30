#!/bin/bash
tasks=("Ackley" "Rastrigin" "Rosenbrock" "Sphere")
v=1
w='1'
b=50
s='1'

models=("GP" "RF")
acfs=("EI" "LCB")


for task in ${tasks[*]}; do
    for model in ${models[*]}; do
        for acf in ${acfs[*]}; do
            python transopt/agent/experiment.py -n $task -v $v -w $w -b $b -m $model -acf $acf -s $s
        done
    done
done
