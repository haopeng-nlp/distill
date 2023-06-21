root=/gscratch/ark/ivyg/fasttext-debias
fname=tdrop0.0-tdrop0.3-dim64
data_path=distill/data-bin/tdrop0.0:distill/data-bin/tdrop0.3

echo ${fname}

SBATCH_PARTITION=gpu-titan
NUM_GPUS=4

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

MAX_TOKENS=4096 #$(python -c "print($FRAGS_PER_GPU*1024)") #CUDA out of memory
UPDATE_FREQ=$(python -c "print(int(8//$NUM_GPUS))")

if [[ $UPDATE_FREQ < 1 ]]; then
    UPDATE_FREQ=1
fi

echo "MAX_TOKENS=$MAX_TOKENS"
echo "UPDATE_FREQ=$UPDATE_FREQ"

echo "data_path=$data_path"
echo "fname=$fname"

read -n 1 -p "verify config (y/n) " ans
echo ""
if [[ $ans != y ]]; then
    exit
fi

nohup \
fairseq-train ${data_path} \
--arch transformer \
--encoder-embed-dim 64 \
--decoder-embed-dim 64 \
--share-all-embeddings \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--lr 7e-4 \
--warmup-init-lr 1e-7 \
--lr-scheduler inverse_sqrt \
--warmup-updates 4000 \
--update-freq $UPDATE_FREQ \
--optimizer adam \
--adam-betas '(0.9, 0.98)' \
--max-tokens $MAX_TOKENS \
--dropout 0.3 \
--distributed-world-size 16 \
--distributed-port 54186 \
--fp16 \
--max-source-positions 10000 \
--max-target-positions 10000 \
--save-dir ${root}/models/${fname}.dist \
--no-epoch-checkpoints \
--seed 1 \
--max-epoch 100 \
--eval-bleu \
--eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
>> ${root}/logs/${fname}.dist.out
