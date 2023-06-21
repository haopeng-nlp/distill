# update_freq x max_tokens * gpus = 1 x 4096 x 8
update_freq=8
wu=4000
lr=7e-4
data=wmt16
wd=0
drop=0.1
td=0.0
path=/gscratch/ark/ivyg/fasttext-debias
fname=sparse
rm -rf ${path}/models/${fname}
rm ${path}/logs/${fname}.out
nohup \
fairseq-train \
distill/data-bin/wmt16_en_de_bpe32k \
--arch distill_wmt_en_de_big \
--share-all-embeddings \
--optimizer adam \
--seed 0 \
--max-epoch 40 \
--adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt \
--criterion distillation \
--label-smoothing 0. \
--warmup-init-lr 1e-07 \
--max-tokens 2048 \
--dropout ${drop} \
--lr ${lr} \
--weight-decay ${wd} \
--warmup-updates ${wu} \
--max-update 350000 \
--save-dir ${path}/models/${fname} \
--no-epoch-checkpoints \
--update-freq ${update_freq} \
--clip-norm 0. \
--distillation 1.0 \
--student-reduction 4 \
--teacher-dropout ${td} \
--finetune-from-model models/wmt16.en-de.joined-dict.transformer/model.pt \
--ddp-backend=no_c10d \
--find-unused-parameters \
--eval-bleu \
--eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
--fp16 \
> ${path}/logs/${fname}.out
