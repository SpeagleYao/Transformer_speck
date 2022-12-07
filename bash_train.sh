#!/usr/bin/env bash

for bs in `echo 1000 5000 10000 20000`
do
        echo "Batch Size:" ${bs}
        python train_gohr.py --batch_size ${bs} > log_bs/trans_bs${bs}_20e.txt
done