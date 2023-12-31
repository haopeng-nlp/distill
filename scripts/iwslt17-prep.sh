### constants

root=/gscratch/ark/ivyg/fasttext-debias
model_dir=$root/models/mbart.cc25.v2
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_lT,lv_lV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_lK,tr_TR,vi_VN,zh_CN
s=en
t=ja
src=en_XX
tgt=ja_XX
lang_pair=$s-$t

spm_encode=$root/distill/scripts/spm_encode.py
spm_decode=$root/distill/scripts/spm_decode.py
bpe_model=$model_dir/sentence.bpe.model

DICT=$model_dir/dict.txt

REPLACE_UNICODE_PUNCT=$root/mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$root/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$root/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl
REMOVE_DIACRITICS=$root/wmt16-scripts/preprocess/remove-diacritics.py
NORMALIZE_ROMANIAN=$root/wmt16-scripts/preprocess/normalise-romanian.py
TOKENIZER=$root/mosesdecoder/scripts/tokenizer/tokenizer.perl

clean_en_text="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -no-escape -l en"
clean_ro_text="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ro | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -no-escape -l ro"

TEXTS=$root/distill/mbart/2017-01-trnted/texts/$s/$t/$lang_pair

noise=tdrop0.0
echo "noise=$noise"

seed=1
echo "seed=$seed"

retain_dropout=false
echo "retain_dropout=$retain_dropout"

bpe_option=mbart # mbart/original_text/generated_text
dict_option=generated_text # original_text/generated_text

read -n 1 -p "verify config (y/n) " ans
echo ""
if [[ $ans != y ]]; then
    exit
fi

data=$root/distill/mbart/data/$lang_pair
data_bin=$root/distill/mbart/data-bin/$lang_pair
DATA=$root/distill/mbart/data/$noise.$seed/$lang_pair
DATA_BIN=$root/distill/mbart/data-bin/$noise.$seed/$lang_pair

mkdir -p $data
mkdir -p $data_bin
mkdir -p $DATA
mkdir -p $DATA_BIN

# Steps:
#   - [once per dataset] plain text to spm (with default bpe model)
#   - [once per dataset] spm to binary (with default dicts)
#   - [once per noise level] generate text with mbart
#   - [once per noise level] preprocess output (remove diacritics, etc.)
#   - learn bpe on ? (original train data/generated train data) (or use the default bpe model)
#   - learn dicts on ? (original train data/generated train data)
#   - apply bpe to train (generated)
#   - apply bpe to valid and test (original)
#   - binarize with dicts

# tail -n+8 $TEXTS/train.tags.$lang_pair.$s | head -n-3 > $TEXTS/train.$src
# tail -n+8 $TEXTS/train.tags.$lang_pair.$t | head -n-3 > $TEXTS/train.$tgt

# xml to plain text
# for xml in $TEXTS/*.xml; do
#     echo $(basename ${xml%.*})
#     python $root/scripts/strip_xml.py --input $xml --output ${xml%.*}
# done

# rm $TEXTS/valid.$src
# for f in $TEXTS/IWSLT17.TED.*.en; do
#     cat $f >> $TEXTS/valid.$src
# done

# rm $TEXTS/valid.$tgt
# for f in $TEXTS/IWSLT17.TED.*.ja; do
#     cat $f >> $TEXTS/valid.$tgt
# done

### plain text to spm

# for split in train valid; do
#     python $spm_encode --model=$bpe_model < $TEXTS/$split.$src > $data/$split.spm.$src
#     python $spm_encode --model=$bpe_model < $TEXTS/$split.$tgt > $data/$split.spm.$tgt
# done

### spm to binary

# fairseq-preprocess \
#     --source-lang $src \
#     --target-lang $tgt \
#     --srcdict $DICT \
#     --tgtdict $DICT \
#     --trainpref $data/train.spm \
#     --validpref $data/valid.spm \
#     --destdir $data_bin \
#     --workers 70

### binary to plain text (generated by teacher with dropout)

for split in valid; do
    dest=$DATA/$split

    cmd="fairseq-generate $data_bin --path $model_dir/model.pt --task translation_from_pretrained_bart --gen-subset $split --max-tokens 1000 -s $src -t $tgt --batch-size 128 --langs $langs --seed $seed"
    if $retain_dropout; then
        cmd="$cmd --retain-dropout"
    fi
    echo $cmd
    nohup $cmd > $dest.out
    
    # https://github.com/facebookresearch/fairseq/issues/1758
    cat $dest.out | grep -P "^S" | cut -f 2- | sed 's/\['$src'\]//g' > $dest.src
    
    cat $dest.out | grep -P "^H" | cut -f 3- | sed 's/\['$tgt'\]//g' > $dest.sys
    cat $dest.out | grep -P "^T" | cut -f 2- | sed 's/\['$tgt'\]//g' > $dest.ref

    sacrebleu -tok 'none' -s 'none' $dest.ref < $dest.sys
done

### plain text to bpe

# clean original valid and test data; train has been cleaned after fairseq-generate
# for split in valid test; do
#     eval "cat $TEXTS/$split.source | $clean_en_text > $TEXTS/$split.source.clean"
#     eval "cat $TEXTS/$split.target | $clean_ro_text > $TEXTS/$split.target.clean"
# done

# if [[ $bpe_option == mbart ]]; then
#     echo "encoding with $bpe_model"
#     python $spm_encode --model=$bpe_model < $DATA/train.src > $DATA/train.spm.$src
#     python $spm_encode --model=$bpe_model < $DATA/train.sys > $DATA/train.spm.$tgt

#     for split in valid test; do
#         python $spm_encode --model=$bpe_model < $TEXTS/$split.source.clean > $DATA/$split.spm.$src
#         python $spm_encode --model=$bpe_model < $TEXTS/$split.target.clean > $DATA/$split.spm.$tgt
#     done
# else
#     BPEROOT=$root/distill/fairseq/subword-nmt/subword_nmt
#     BPE_TOKENS=32000
#     BPE_CODE=$model_dir/$lang_pair.code
#     TRAIN=$data/train.$lang_pair
#     rm $TRAIN
#     if [[ $bpe_option == original_text ]]; then
#         eval "cat $TEXTS/train.source | $clean_en_text >> $TRAIN"
#         eval "cat $TEXTS/train.target | $clean_ro_text >> $TRAIN"
#     elif [[ $bpe_option == generated_text ]]; then
#         eval "cat $DATA/train.src | $clean_en_text >> $TRAIN"
#         eval "cat $DATA/train.sys | $clean_ro_text >> $TRAIN"
#     else
#         echo "$bpe_option not recognized"
#         exit
#     fi

#     echo "learn_bpe.py on $TRAIN..."
#     python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

#     echo "apply_bpe.py.."
#     python $BPEROOT/apply_bpe.py -c $BPE_CODE < $DATA/train.src > $DATA/bpe.train.$src
#     python $BPEROOT/apply_bpe.py -c $BPE_CODE < $DATA/train.sys > $DATA/bpe.train.$tgt

#     for split in valid test; do
#         python $BPEROOT/apply_bpe.py -c $BPE_CODE < $TEXTS/$split.source.clean > $DATA/bpe.$split.$src
#         python $BPEROOT/apply_bpe.py -c $BPE_CODE < $TEXTS/$split.target.clean > $DATA/bpe.$split.$tgt
#     done
# fi

# ### bpe to binary

# if [[ $dict_option == original_text ]]; then
#     echo "not yet implemented"
#     exit
# elif [[ $dict_option == generated_text ]]; then
#     if [[ $bpe_option == mbart ]]; then
#         echo "*.spm"
#         trainpref=train.spm
#         validpref=valid.spm
#         testpref=test.spm
#         destdir=$DATA_BIN/spm
#         dictdir=$root/distill/mbart/data-bin/tdrop0.0/$lang_pair/spm
#     else
#         echo "bpe.*"
#         trainpref=bpe.train
#         validpref=bpe.valid
#         testpref=bpe.test
#         destdir=$DATA_BIN/bpe
#         dictdir=$root/distill/mbart/data-bin/tdrop0.0/$lang_pair/bpe
#     fi

#     echo "creating $destdir"
#     mkdir -p $destdir
#     rm $destdir/dict.* $destdir/preprocess.log

#     if [[ $noise == tdrop0.0 ]]; then
#         echo "new dicts"
#         fairseq-preprocess \
#             --joined-dictionary \
#             --source-lang $src \
#             --target-lang $tgt \
#             --trainpref $DATA/$trainpref \
#             --validpref $DATA/$validpref \
#             --testpref  $DATA/$testpref \
#             --destdir $destdir \
#             --workers 70
#     else
#         echo "existing dicts"
#         fairseq-preprocess \
#             --srcdict $dictdir/dict.en_XX.txt \
#             --tgtdict $dictdir/dict.ro_RO.txt \
#             --source-lang $src \
#             --target-lang $tgt \
#             --trainpref $DATA/$trainpref \
#             --validpref $DATA/$validpref \
#             --testpref  $DATA/$testpref \
#             --destdir $destdir \
#             --workers 70
#     fi
# else
#     echo "$dict_option not recognized"
#     exit
# fi
# ### constants

# root=/gscratch/ark/ivyg/fasttext-debias
# model_dir=$root/models/MBART_finetuned_enro
# langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_lT,lv_lV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_lK,tr_TR,vi_VN,zh_CN
# src=en_XX
# tgt=ro_RO
# lang_pair=ro-en

# spm_encode=$root/distill/scripts/spm_encode.py
# spm_decode=$root/distill/scripts/spm_decode.py
# bpe_model=$model_dir/sentence.bpe.model

# DICT=$model_dir/dict.txt

# REPLACE_UNICODE_PUNCT=$root/mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl
# NORM_PUNC=$root/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl
# REM_NON_PRINT_CHAR=$root/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl
# REMOVE_DIACRITICS=$root/wmt16-scripts/preprocess/remove-diacritics.py
# NORMALIZE_ROMANIAN=$root/wmt16-scripts/preprocess/normalise-romanian.py
# TOKENIZER=$root/mosesdecoder/scripts/tokenizer/tokenizer.perl
# # LC=$root/mosesdecoder/scripts/tokenizer/lowercase.perl

# clean_en_text="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -no-escape -l en"
# clean_ro_text="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ro | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -no-escape -l ro"

# TEXTS=$root/distill/mbart/wmt_en_ro

# noise=tdrop0.3
# echo "noise=$noise"

# seed=0
# echo "seed=$seed"

# retain_dropout=true
# echo "retain_dropout=$retain_dropout"

# bpe_option=mbart # mbart/original_text/generated_text
# dict_option=generated_text # original_text/generated_text

# read -n 1 -p "verify config (y/n) " ans
# echo ""
# if [[ $ans != y ]]; then
#     exit
# fi

# data=$root/distill/mbart/data/$lang_pair
# data_bin=$root/distill/mbart/data-bin/$lang_pair
# DATA=$root/distill/mbart/data/$noise.$seed/$lang_pair
# DATA_BIN=$root/distill/mbart/data-bin/$noise.$seed/$lang_pair

# mkdir -p $data
# mkdir -p $data_bin
# mkdir -p $DATA
# mkdir -p $DATA_BIN

# # Steps:
# #   - [once per dataset] plain text to spm (with default bpe model)
# #   - [once per dataset] spm to binary (with default dicts)
# #   - [once per noise level] generate text with mbart
# #   - [once per noise level] preprocess output (remove diacritics, etc.)
# #   - learn bpe on ? (original train data/generated train data) (or use the default bpe model)
# #   - learn dicts on ? (original train data/generated train data)
# #   - apply bpe to train (generated)
# #   - apply bpe to valid and test (original)
# #   - binarize with dicts

# # sgml to plain text (for .sgm files)
# # python $root/scripts/strip_sgml.py --input INPUT --output OUTPUT

# ### plain text to spm

# # for split in train valid test; do
# #     python $spm_encode --model=$bpe_model < $TEXTS/$split.source > $data/$split.spm.$src
# #     python $spm_encode --model=$bpe_model < $TEXTS/$split.target > $data/$split.spm.$tgt
# # done

# ### spm to binary

# # fairseq-preprocess \
# #     --source-lang $src \
# #     --target-lang $tgt \
# #     --srcdict $DICT \
# #     --tgtdict $DICT \
# #     --trainpref $data/train.spm \
# #     --validpref $data/valid.spm \
# #     --testpref $data/test.spm \
# #     --destdir $data_bin \
# #     --workers 70

# ### binary to plain text (generated by teacher with dropout)

# # for split in train; do
# #     dest=$DATA/$split

# #     cmd="fairseq-generate $data_bin --path $model_dir/model.pt --task translation_from_pretrained_bart --gen-subset $split --max-tokens 1000 -s $src -t $tgt --batch-size 128 --langs $langs --seed $seed"
# #     if $retain_dropout; then
# #         cmd="$cmd --retain-dropout"
# #     fi
# #     echo $cmd
# #     nohup $cmd > $dest.out
    
# #     # https://github.com/facebookresearch/fairseq/issues/1758
# #     cat $dest.out | grep -P "^S" | cut -f 2- | sed 's/\['$src'\]//g' > $dest.src.temp
# #     eval "$spm_decode --model=$bpe_model --input=$dest.src.temp | $clean_en_text > $dest.src"
    
# #     cat $dest.out | grep -P "^H" | cut -f 3- | sed 's/\['$tgt'\]//g' > $dest.sys.temp
# #     cat $dest.out | grep -P "^T" | cut -f 2- | sed 's/\['$tgt'\]//g' > $dest.ref.temp
# #     for file in $dest.sys $dest.ref; do
# #         eval "$spm_decode --model=$bpe_model --input=$file.temp | $clean_ro_text > $file"
# #     done
# #     sacrebleu -tok 'none' -s 'none' $dest.ref < $dest.sys
# #     rm $DATA/*.temp
# # done

# ### plain text to bpe

# # clean original valid and test data; train has been cleaned after fairseq-generate
# # for split in valid test; do
# #     eval "cat $TEXTS/$split.source | $clean_en_text > $TEXTS/$split.source.clean"
# #     eval "cat $TEXTS/$split.target | $clean_ro_text > $TEXTS/$split.target.clean"
# # done

# if [[ $bpe_option == mbart ]]; then
#     echo "encoding with $bpe_model"
#     python $spm_encode --model=$bpe_model < $DATA/train.src > $DATA/train.spm.$src
#     python $spm_encode --model=$bpe_model < $DATA/train.sys > $DATA/train.spm.$tgt

#     for split in valid test; do
#         python $spm_encode --model=$bpe_model < $TEXTS/$split.source.clean > $DATA/$split.spm.$src
#         python $spm_encode --model=$bpe_model < $TEXTS/$split.target.clean > $DATA/$split.spm.$tgt
#     done
# else
#     BPEROOT=$root/distill/fairseq/subword-nmt/subword_nmt
#     BPE_TOKENS=32000
#     BPE_CODE=$model_dir/$lang_pair.code
#     TRAIN=$data/train.$lang_pair
#     rm $TRAIN
#     if [[ $bpe_option == original_text ]]; then
#         eval "cat $TEXTS/train.source | $clean_en_text >> $TRAIN"
#         eval "cat $TEXTS/train.target | $clean_ro_text >> $TRAIN"
#     elif [[ $bpe_option == generated_text ]]; then
#         eval "cat $DATA/train.src | $clean_en_text >> $TRAIN"
#         eval "cat $DATA/train.sys | $clean_ro_text >> $TRAIN"
#     else
#         echo "$bpe_option not recognized"
#         exit
#     fi

#     echo "learn_bpe.py on $TRAIN..."
#     python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

#     echo "apply_bpe.py.."
#     python $BPEROOT/apply_bpe.py -c $BPE_CODE < $DATA/train.src > $DATA/bpe.train.$src
#     python $BPEROOT/apply_bpe.py -c $BPE_CODE < $DATA/train.sys > $DATA/bpe.train.$tgt

#     for split in valid test; do
#         python $BPEROOT/apply_bpe.py -c $BPE_CODE < $TEXTS/$split.source.clean > $DATA/bpe.$split.$src
#         python $BPEROOT/apply_bpe.py -c $BPE_CODE < $TEXTS/$split.target.clean > $DATA/bpe.$split.$tgt
#     done
# fi

# ### bpe to binary

# if [[ $dict_option == original_text ]]; then
#     echo "not yet implemented"
#     exit
# elif [[ $dict_option == generated_text ]]; then
#     if [[ $bpe_option == mbart ]]; then
#         echo "*.spm"
#         trainpref=train.spm
#         validpref=valid.spm
#         testpref=test.spm
#         destdir=$DATA_BIN/spm
#         dictdir=$root/distill/mbart/data-bin/tdrop0.0/$lang_pair/spm
#     else
#         echo "bpe.*"
#         trainpref=bpe.train
#         validpref=bpe.valid
#         testpref=bpe.test
#         destdir=$DATA_BIN/bpe
#         dictdir=$root/distill/mbart/data-bin/tdrop0.0/$lang_pair/bpe
#     fi

#     echo "creating $destdir"
#     mkdir -p $destdir
#     rm $destdir/dict.* $destdir/preprocess.log

#     if [[ $noise == tdrop0.0 ]]; then
#         echo "new dicts"
#         fairseq-preprocess \
#             --joined-dictionary \
#             --source-lang $src \
#             --target-lang $tgt \
#             --trainpref $DATA/$trainpref \
#             --validpref $DATA/$validpref \
#             --testpref  $DATA/$testpref \
#             --destdir $destdir \
#             --workers 70
#     else
#         echo "existing dicts"
#         fairseq-preprocess \
#             --srcdict $dictdir/dict.en_XX.txt \
#             --tgtdict $dictdir/dict.ro_RO.txt \
#             --source-lang $src \
#             --target-lang $tgt \
#             --trainpref $DATA/$trainpref \
#             --validpref $DATA/$validpref \
#             --testpref  $DATA/$testpref \
#             --destdir $destdir \
#             --workers 70
#     fi
# else
#     echo "$dict_option not recognized"
#     exit
# fi
