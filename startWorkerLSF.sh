#!/bin/bash
#BSUB -o dasksched.txt
#BSUB -e dasksched-error.txt
#BSUB -M 1000000
#BSUB -R 1000

pyenv activate myenv
dask-worker --scheduler-file /home/dmoi/scheduler.json

