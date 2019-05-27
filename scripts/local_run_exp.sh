#!/bin/sh
NAME=exp_`date "+%m_%d_%H_%M_%S"`
ALGO=$1
EXP_FILE=$2
NUM_WORKERS=200

python -m es_distributed.main master --algo "$ALGO" --num_workers "$NUM_WORKERS" --exp_file "$EXP_FILE"
