root=/net/nfs.cirrascale/allennlp/haop/fasttext-debias
source $root/scripts/const.sh

bpe_model=$root/models/mbart.cc25.v2/sentence.bpe.model
DICT=$root/models/mbart.cc25.v2/dict.txt

split=test
lang_pair=en-tr
dim=
if [[ $lang_pair == en-tr ]]; then
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

test_set=wmt16
if [ $lang_pair == en-tr ] || [ $lang_pair == tr-en ]; then
    test_set=wmt17
fi

# for noise in $root/models/mbart/checkpoints/en-tr/*/; do
# noise=$(basename "$noise")
for noise in batch-order-no-gold; do
echo $root
echo "@@@ $noise @@@"

bin=$root/distill/mbart/data-bin/gold/$lang_pair/spm
if [ $noise == mbart ]; then
    bin=$root/distill/mbart/data-bin/$lang_pair
fi
model_ckpt=models/mbart/$lang_pair/$noise$dim.dist/checkpoint_best.pt
if [ $noise == mbart ]; then
    if [ $lang_pair == en-tr ]; then
        model_ckpt=models/mbart-entr/model.pt
    elif [ $lang_pair == tr-en ]; then
        model_ckpt=models/mbart-tren/model.pt
    elif [ $lang_pair == en-de ]; then
        model_ckpt=models/mbart-ende/model.pt
    elif [ $lang_pair == de-en ]; then
        model_ckpt=models/mbart-deen/model.pt
    fi
fi
echo $root
dest_dir=$root/distill/mbart/eval/$lang_pair/$noise$dim
dest=$dest_dir/$split
echo $dest_dir
mkdir -p $dest_dir

if [ $noise == mbart ]; then
    # nohup \
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
    # nohup \
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
$spm_decode --model=$bpe_model --input=$dest.sys.temp > $dest.sys

perl $DETOKENIZER -l ${tgt:0:2} < $dest.sys > $dest.sys.detok
if [ $split == test ]; then
    cat $dest.sys.detok | sacrebleu -t $test_set -l $lang_pair
else
    cat $dest.out | grep -P "^T" | sort -V | cut -f 2- | sed 's/\['$tgt'\]//g' > $dest.ref.temp
    $spm_decode --model=$bpe_model --input=$dest.ref.temp > $dest.ref

    perl $DETOKENIZER -l ${tgt:0:2} < $dest.ref > $dest.ref.detok
    cat $dest.sys.detok | sacrebleu -tok none -s none -b $dest.ref.detok
fi

# rm $dest_dir/*.temp

done
