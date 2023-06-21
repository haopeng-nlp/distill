root=/net/nfs.cirrascale/allennlp/haop/fasttext-debias
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_lT,lv_lV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_lK,tr_TR,vi_VN,zh_CN

spm_encode=$root/distill/scripts/spm_encode.py
spm_decode=$root/distill/scripts/spm_decode.py

DATA=$root/distill/mbart/data/$noise/$lang_pair
DATA_BIN=$root/distill/mbart/data-bin/$noise/$lang_pair

REPLACE_UNICODE_PUNCT=$root/mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$root/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$root/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl
REMOVE_DIACRITICS=$root/wmt16-scripts/preprocess/remove-diacritics.py
NORMALIZE_ROMANIAN=$root/wmt16-scripts/preprocess/normalise-romanian.py
TOKENIZER=$root/mosesdecoder/scripts/tokenizer/tokenizer.perl
DETOKENIZER=$root/mosesdecoder/scripts/tokenizer/detokenizer.perl
# LC=$root/mosesdecoder/scripts/tokenizer/lowercase.perl
