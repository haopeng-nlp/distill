#!/bin/bash

# for eta in 1 2 4 8 16
# do
#         python src/main.py --model bert --use-pretrained --objective joint \
#                 --train-data=data/biasbios/train.pickle --dev-data=data/biasbios/dev.pickle \
#                 --weight-decay 0.0 --dropout 0.3 --eta $eta --lr 0.01  \
#                 --batch-size 500 --warmup-updates 100
# done

# python src/main.py --model bert --objective acc --num-epochs 2 \
#         --train-data=data/biasbios/train.pickle --dev-data=data/biasbios/dev.pickle \
#         --weight-decay 0.0 --dropout 0.3 --eta 1 --lr 1e-4  \
#         --batch-size 32 --warmup-updates 100 --log-per-updates 500

# for eta in 4 6 8
# do
#         python src/main.py --model bert --objective joint --num-epochs 2 \
#                 --train-data=data/biasbios/train.pickle --dev-data=data/biasbios/dev.pickle \
#                 --weight-decay 0.0 --dropout 0.3 --eta $eta --lr 1e-4  \
#                 --batch-size 32 --warmup-updates 100 --log-per-updates 500
# done

# python src/main.py --model bert --dataset amzn --objective acc --num-epochs 20 \
#         --train-data=data/amzn/train.pickle --dev-data=data/amzn/dev.pickle --test-data=data/amzn/test.pickle \
#         --weight-decay 0.0 --dropout 0.3 --eta 32 --lr 1e-4  \
#         --batch-size 32 --warmup-updates 100

python src/main.py \
	--model bert \
	--dataset amzn \
	--objective label_smoothing \
	--lamb 0.1 \
	--num-epochs 30 \
    --data-path="data/amzn" \
	--save-path="model/amzn" \
    --weight-decay 0.0 \
	--dropout 0.3 \
	--eta 1 \
	--lr 5e-5  \
    --batch-size 32 \
	--warmup-updates 100 \
	--source 8 \
	--target 19
