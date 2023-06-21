root=/gscratch/ark/ivyg/fasttext-debias
source $root/scripts/const.sh

bpe_model=$root/models/mbart.cc25.v2/sentence.bpe.model
DICT=$root/models/mbart.cc25.v2/dict.txt

split=test
lang_pair=ro-en
dim=-d128
if [[ $lang_pair == en-ro ]]; then
    src=en_XX
    tgt=ro_RO
elif [[ $lang_pair == ro-en ]]; then
    src=ro_RO
    tgt=en_XX
else
    echo "invalid lang_pair"
    exit
fi

for noise in gold tdrop0.0 tdrop0.3 tdrop0.0-tdrop0.3; do

echo "@@@ $noise @@@"

bin=$root/distill/mbart/data-bin/gold/$lang_pair/spm
if [ $noise == mbart ]; then
    bin=$root/distill/mbart/data-bin/$lang_pair
fi
model_ckpt=models/mbart/$lang_pair/$noise$dim.dist/checkpoint_best.pt
if [ $noise == mbart ]; then
    if [ $lang_pair == en-ro ]; then
        model_ckpt=models/mbart-enro/model.pt
    elif [ $lang_pair == ro-en ]; then
        model_ckpt=models/mbart-roen/model.pt
    fi
fi

dest_dir=$root/distill/mbart/eval/$lang_pair/$noise$dim
dest=$dest_dir/$split

mkdir -p $dest_dir

if [ $noise == mbart ]; then
    nohup \
    fairseq-generate $bin \
        --path $model_ckpt \
        --task translation_from_pretrained_bart \
        --gen-subset $split \
        --max-tokens 1000 \
        --batch-size 32 \
        -s $src -t $tgt \
        --langs $langs \
    > $dest.out
else
    nohup \
    fairseq-generate $bin \
        --path $model_ckpt \
        --gen-subset $split \
        --max-tokens 1000 \
        --batch-size 32 \
        --source-lang $src \
        --target-lang $tgt \
    > $dest.out
fi

cat $dest.out | grep -P "^H" | sort -V | cut -f 3- | sed 's/\['$tgt'\]//g' > $dest.sys.temp
cat $dest.out | grep -P "^T" | sort -V | cut -f 2- | sed 's/\['$tgt'\]//g' > $dest.ref.temp

# for file in $dest.sys $dest.ref; do
#     $spm_decode --model=$bpe_model --input=$file.temp \
#     | $REPLACE_UNICODE_PUNCT \
#     | $NORM_PUNC -l ${tgt:0:2} \
#     | $REM_NON_PRINT_CHAR \
#     | $NORMALIZE_ROMANIAN \
#     | $REMOVE_DIACRITICS \
#     | $TOKENIZER -no-escape -l ${tgt:0:2} \
#     > $file

# sacrebleu -tok 'none' -s 'none' $dest.ref < $dest.sys

# done

$spm_decode --model=$bpe_model --input=$dest.sys.temp \
| $REPLACE_UNICODE_PUNCT \
| $NORM_PUNC -l ${tgt:0:2} \
| $REM_NON_PRINT_CHAR \
| $NORMALIZE_ROMANIAN \
> $dest.sys

perl $DETOKENIZER -l ${tgt:0:2} < $dest.sys > $dest.sys.detok
cat $dest.sys.detok | sacrebleu -t wmt16 -l $lang_pair

# rm $dest_dir/*.temp

done
