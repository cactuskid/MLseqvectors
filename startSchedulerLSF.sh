#!/bin/bash

#BSUB -o dasksched.txt
#BSUB -e dasksched-error.txt
#BSUB -M 100000
#BSUB -R 100

pyenv activate myenv
dask-scheduler --scheduler-file /home/dmoi/scheduler.json

