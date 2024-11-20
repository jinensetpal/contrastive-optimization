#!/usr/bin/bash

curl -LO https://dagshub.com/jinensetpal/contrastive-optimization/src/main/s3:/contrastive-optimization/imagenet.tgz
tar xvzf imagenet.tgz -C data/
