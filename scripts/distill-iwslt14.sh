# https://github.com/facebookresearch/fairseq/blob/b5a039c292facba9c73f59ff34621ec131d82341/fairseq/models/transformer/transformer_legacy.py#L224-L234
# @register_model_architecture("transformer", "transformer_iwslt_de_en")
# def transformer_iwslt_de_en(args):
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
#     args.encoder_layers = getattr(args, "encoder_layers", 6)
#     args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
#     args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
#     args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
#     args.decoder_layers = getattr(args, "decoder_layers", 6)
#     base_architecture(args)

coef=0.0

fairseq-train distill/data-bin/iwslt14.32k.en-de \
    --arch distill_iwslt_de_en \
    --encoder-embed-dim 1024 \
    --decoder-embed-dim 1024 \
    --max-epoch 40 \
    --share-all-embeddings \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion distillation \
    --distillation $coef \
    --max-tokens 4096 \
    --update-freq 1 \
    --no-epoch-checkpoints \
    --save-dir model/distill$coef-new \
    --ddp-backend=no_c10d \
    --find-unused-parameters \
    --student-reduction 4 \
    --teacher-dropout 0.3 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --finetune-from-model model/wmt16.en-de.joined-dict.transformer/model.pt \
    --tensorboard-logdir logs/distill$coef-new
