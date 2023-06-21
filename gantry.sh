gantry run \
	--beaker-image haop/distill \
	--gpus 2 \
	--allow-dirty \
	--workspace ai2/distillation \
	--priority high \
	--cluster ai2/*-cirrascale -- \
	bash scripts/mbart-prep-data.sh $1 $2
