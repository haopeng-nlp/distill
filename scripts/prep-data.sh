root=/gscratch/ark/ivyg/fasttext-debias
model=${root}/models/mbart.cc25.v2/model.pt ###
noise=tdrop0.0 ###
outdir=${root}/distill/data/${noise}
src=en
tgt=de

echo $noise

# mkdir -p ${root}/distill/data/${noise}
# mkdir -p ${root}/distill/data-bin/${noise}

# ### --retain-dropout AND --seed ###
# for split in train; do
#     dest=${outdir}/${split}.out
#     rm ${dest}
#     echo "fairseq-generate ${split}"
#     nohup \
#     fairseq-generate distill/data-bin/wmt16_en_de_bpe32k \
#         --path ${model} \
#         --gen-subset ${split} \
#         --batch-size 128 \
#         --remove-bpe \
#         --retain-dropout \
#         --seed 1 \
#     > ${dest}
#     if [ ${split} == train ]; then
#         grep --text ^S ${dest} | cut -f2- > ${outdir}/${split}.en
#         grep --text ^H ${dest} | cut -f3- > ${outdir}/${split}.de
#     else
#         echo "grepping sorted output for ${split}"
#         # https://github.com/facebookresearch/fairseq/issues/2036
#         grep --text ^S ${dest} | LC_ALL=C sort -V | cut -f2- > ${outdir}/${split}.en
#         grep --text ^H ${dest} | LC_ALL=C sort -V | cut -f3- > ${outdir}/${split}.de
#     fi
# done

BPEROOT=${root}/distill/fairseq/subword-nmt/subword_nmt
BPE_CODE=${root}/models/wmt16.en-de.joined-dict.transformer/bpecodes
for L in $src $tgt; do
    echo "apply_bpe.py to train.$L..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < ${outdir}/train.$L > ${outdir}/bpe.train.$L
done

echo "preprocessing bpe.train"
fairseq-preprocess --source-lang en --target-lang de \
    --srcdict ${root}/models/wmt16.en-de.joined-dict.transformer/dict.en.txt \
    --tgtdict ${root}/models/wmt16.en-de.joined-dict.transformer/dict.de.txt \
    --trainpref ${outdir}/bpe.train \
    --destdir ${root}/distill/data-bin/${noise} \
    --workers 20

bin=/gscratch/ark/ivyg/fasttext-debias/distill/data-bin
echo "copying $(find ${bin}/wmt16_en_de_bpe32k/valid.* -printf "%f ")to ${bin}/${noise}"
cp ${bin}/wmt16_en_de_bpe32k/valid.* ${bin}/${noise}/
echo "copying $(find ${bin}/wmt16_en_de_bpe32k/test.* -printf "%f ")to ${bin}/${noise}"
cp ${bin}/wmt16_en_de_bpe32k/test.*  ${bin}/${noise}/
# cmp --silent $old $new || echo "files are different"
