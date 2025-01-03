#!/usr/bin/bash

run=`ls .log/  | wc -l`
echo "Writing to .log/training-$run.log"

MLFLOW_TRACKING_USERNAME=jinensetpal \
MLFLOW_TRACKING_PASSWORD=$MLFLOW_TOKEN \
OMP_NUM_THREADS=80 \
nohup torchrun --standalone --nnodes=1 --nproc_per_node=gpu -m src.model.train contrastive nocheckpoint > .log/training-$run.log 2>&1 &
