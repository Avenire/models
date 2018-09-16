SEED=$1
DATASET=fsns
MODEL_ID=$2
PROJ_DIR=~/$DATASET"_"$MODEL_ID
STOP=400000
EVAL_EVERY=10000
[[ $4 > 0 ]] && GPU=$4 || GPU=0
BATCH=16
./run_experiment.sh $GPU $MODEL_ID $PROJ_DIR $DATASET $EVAL_EVERY $STOP $EVAL_EVERY $BATCH $SEED 2



