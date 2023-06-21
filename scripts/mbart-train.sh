#!/bin/bash

SBATCH_PARTITION=gpu-2080ti
NUM_GPUS=6

echo "SBATCH_PARTITION=$SBATCH_PARTITION"
echo "NUM_GPUS=$NUM_GPUS"

read -n 1 -p "verify config (y/n) " ans
echo ""
if [[ $ans != y ]]; then
    exit
fi

if [[ $SBATCH_PARTITION == "gpu-2080ti" || $SBATCH_PARTITION == "" ]]; then
    FRAGS_PER_GPU=4
elif [[ $SBATCH_PARTITION == "gpu-titan" || $SBATCH_PARTITION == "gpu-rtx6k" ]]; then
    FRAGS_PER_GPU=16
elif [[ $SBATCH_PARTITION == "gpu-a40" ]]; then
    FRAGS_PER_GPU=32
else
    echo "wrong partition"
fi

MAX_TOKENS=1024 #$(python -c "print($FRAGS_PER_GPU*1024)") #CUDA out of memory
UPDATE_FREQ=$(python -c "print(int(32//$NUM_GPUS))")

if [[ $UPDATE_FREQ < 1 ]]; then
    UPDATE_FREQ=1
fi

echo "MAX_TOKENS=$MAX_TOKENS"
echo "UPDATE_FREQ=$UPDATE_FREQ"

root=/gscratch/ark/ivyg/fasttext-debias
seed=1
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_lT,lv_lV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_lK,tr_TR,vi_VN,zh_CN

###
lang_pair=en-ro
dim=512
###

if [[ $lang_pair == en-ro ]]; then
    src=en_XX
    tgt=ro_RO
elif [[ $lang_pair == ro-en ]]; then
    src=ro_RO
    tgt=en_XX
elif [[ $lang_pair == en-tr ]]; then
    src=en_XX
    tgt=tr_TR
elif [[ $lang_pair == tr-en ]]; then
    src=tr_TR
    tgt=en_XX
elif [[ $lang_pair == en-de ]]; then
    src=en_XX
    tgt=de_DE
elif [[ $lang_pair == de-en ]]; then
    src=de_DE
    tgt=en_XX
else
    echo "invalid lang_pair"
    exit
fi

for noise in ordered-epoch; do

bin=$root/distill/mbart/data-bin/$noise/$lang_pair/spm
if [ $dim == 512 ]; then
    fname=$noise
    drop=0.3
else
    fname=$noise-d$dim
    drop=0.1
fi

# if [ $noise == ordered-epoch-no-gold ]; then
#     bin=$root/distill/mbart/data-bin/tdrop0.0/$lang_pair/spm:$root/distill/mbart/data-bin/tdrop0.1/$lang_pair/spm:$root/distill/mbart/data-bin/tdrop0.2/$lang_pair/spm:$root/distill/mbart/data-bin/tdrop0.3/$lang_pair/spm
# fi
if [ $noise == ordered-epoch ]; then
    bin=$root/distill/mbart/data-bin/gold/$lang_pair/spm:$root/distill/mbart/data-bin/tdrop0.0/$lang_pair/spm:$root/distill/mbart/data-bin/tdrop0.1/$lang_pair/spm:$root/distill/mbart/data-bin/tdrop0.2/$lang_pair/spm:$root/distill/mbart/data-bin/tdrop0.3/$lang_pair/spm
fi

echo "bin=$bin"
echo "fname=$fname"

# read -n 1 -p "verify config (y/n) " ans
# echo ""
# if [[ $ans != y ]]; then
#     exit
# fi

mkdir -p $root/logs/mbart/$lang_pair
mkdir -p $root/models/mbart/$lang_pair
rm $root/logs/mbart/$lang_pair/$fname.dist.out
rm -rf $root/models/mbart/$lang_pair/$fname.dist

nohup \
fairseq-train $bin \
    --arch transformer \
    --encoder-embed-dim $dim \
    --decoder-embed-dim $dim \
    --update-freq $UPDATE_FREQ \
    --save-dir $root/models/mbart/$lang_pair/$fname.dist \
    --no-epoch-checkpoints \
    --keep-best-checkpoints 5 \
    --activation-fn "gelu" \
    --share-all-embeddings \
    --optimizer adam \
    --seed 42 \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --warmup-init-lr 1e-07 \
    --max-tokens $MAX_TOKENS \
    --lr 7e-4 \
    --weight-decay 0 \
    --dropout $drop \
    --max-update 50000 \
    --warmup-updates 4000 \
    --clip-norm 0 \
    --fp16 \
    --fp16-init-scale 16 \
    --fp16-scale-window 1024 \
    --ddp-backend=no_c10d \
    --find-unused-parameters \
    --eval-bleu \
    --eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
> $root/logs/mbart/$lang_pair/$fname.dist.out

done
