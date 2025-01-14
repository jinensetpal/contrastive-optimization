#!/usr/bin/env python3

from src import const
from glob import glob
import mlflow
import json

if __name__ == '__main__':
    mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)

    for checkpoint_metadata in glob((const.MODELS_DIR / '**' / 'checkpoint_metadata.json').as_posix()):
        try:
            with mlflow.start_run(json.load(open(checkpoint_metadata))['mlflow_run_id']):
                mlflow.set_tag('dagshub.labels.latest', '-')
        except Exception:
            print(checkpoint_metadata, 'failed')
