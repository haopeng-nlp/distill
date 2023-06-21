root=/gscratch/ark/ivyg/fasttext-debias
fname=tdrop0.3-seed42
model_dir=$root/models/$fname.dist
split=test
outdir=$root/distill/data/temp
mkdir -p $outdir
dest=$outdir/$split.out
fairseq-generate distill/data-bin/wmt16_en_de_bpe32k \
    --path $model_dir/checkpoint_last.pt \
    --gen-subset $split \
    --batch-size 128 \
    --remove-bpe \
    --seed 1 \
> $dest

grep --text ^H $dest | cut -f3- > $outdir/$split.sys
grep --text ^T $dest | cut -f2- > $outdir/$split.ref

wc -l $outdir/$split.sys
wc -l $outdir/$split.ref

fairseq-score -s $outdir/$split.sys -r $outdir/$split.ref --ignore-case

rm -rf $outdir
