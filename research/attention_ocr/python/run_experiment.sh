DEVICE=$1
MODEL_ID="$2"
BASE_LOG_DIR=$3
DATASET=$4
START_EVAL_AFTER_STEP=$5
END_EXPERIMENT_STEP=$6
STEPS_TO_TRAIN_PER_EVAL=$7
BATCH=$8
SEED=$9
PURGE=${10}

[[ $PURGE > 0 ]] || PURGE=0
[[ $BATCH > 0 ]] || BATCH=32
[[ $SEED > 0 ]] || SEED=123456

NAME=$MODEL_ID"_"$SEED
TRAIN_DIR=$BASE_LOG_DIR'/train/'"$NAME"
EVAL_DIR=$BASE_LOG_DIR'/eval/'"$NAME"
VALID_DIR=$BASE_LOG_DIR'/valid/'"$NAME"
export CUDA_VISIBLE_DEVICES=""$DEVICE""

echo "WILL SAVE TRAIN DATA TO: " $TRAIN_DIR &&
mkdir -p $TRAIN_DIR &&
echo "WILL SAVE EVAL DATA TO: " $EVAL_DIR &&
mkdir -p $EVAL_DIR
echo "CUDA VISIBLE DEVICES" $CUDA_VISIBLE_DEVICES
echo "BATCH: "$BATCH
echo "SEED: "$SEED
echo "VISIBLE DEVICES: "$CUDA_VISIBLE_DEVICES
echo "KEEPS LATEST: "$PURGE "checkpoint folders"
echo "MODEL_ID=$MODEL_ID"

set -o pipefail
BACK_OFF=1
MAX_BACKOFF=64
python copy_fresh_ckpt.py --train_dir=$TRAIN_DIR --purge_stale_thresh=$PURGE
LATEST=$(cat $TRAIN_DIR/latest_ckpt)
echo $LATEST
if [[ $LATEST = -1 ]] || [[ $LATEST -ge $END_EXPERIMENT_STEP ]]; then
echo "EXPERIMENT ALREADY FINISHED."
exit 0
fi
[[ $LATEST > 0 ]] && LATEST=$(($LATEST+$STEPS_TO_TRAIN_PER_EVAL)) || LATEST=$START_EVAL_AFTER_STEP

echo "WILL EVAL EVERY "$STEPS_TO_TRAIN_PER_EVAL" STEPS UNTIL STEP "$END_EXPERIMENT_STEP" STARTING ON "$LATEST" STEP"
for (( i=$LATEST; i<=$END_EXPERIMENT_STEP; i += $STEPS_TO_TRAIN_PER_EVAL ))
  do

# TRAIN
python train.py --dataset_name="$DATASET" --use_attention=true --use_autoregression=true --seed_base=$SEED \
--use_encoding=true --model_id="$MODEL_ID" \
--train_log_dir="$TRAIN_DIR" --batch_size=$BATCH --max_number_of_steps=$i \
2>&1 | tee -a $TRAIN_DIR/train.log

# CHECK STATUS CODE
if ! $(exit $?); then
    echo "[run_experiment.sh] Train failed at step: "$i
    ~/health_check.sh "Train failed for experiment $TRAIN_DIR at step $i"
    python copy_fresh_ckpt.py --train_dir=$TRAIN_DIR --purge_stale_thresh=$PURGE
    i=$(($i-$STEPS_TO_TRAIN_PER_EVAL))
    [[ $BACK_OFF -lt $MAX_BACKOFF ]] && BACK_OFF=$(($BACK_OFF * 2))
    echo $BACK_OFF
    echo $MAX_BACKOFF
    sleep $BACK_OFF
    echo "Retrying after $BACK_OFF..."
    continue
fi
BACK_OFF=1
SAVED_CHKPT_DIR=$TRAIN_DIR/checkpoint_$i
# SAVE INTERMEDIATE CHECKPOINT
mkdir -p $SAVED_CHKPT_DIR && cp $TRAIN_DIR/model.ckpt-* $SAVED_CHKPT_DIR/ && cp $TRAIN_DIR/checkpoint $SAVED_CHKPT_DIR &&
# CHECK STATUS CODE
if ! $(exit $?); then
    echo "[run_experiment.sh] Move checkpoints failed at step: "$i
fi

python copy_fresh_ckpt.py --train_dir=$TRAIN_DIR --purge_stale_thresh=$PURGE
python 'eval.py' --dataset_name="$DATASET" --use_attention=true --use_autoregression=true \
--use_encoding=true --model_id="$MODEL_ID" \
--split_name=eval_full \
--train_log_dir="$TRAIN_DIR" --batch_size=$BATCH --eval_log_dir="$EVAL_DIR" 2>&1 | tee -a $EVAL_DIR"/eval.log"

# CHECK STATUS CODE
if ! $(exit $?); then
    echo "[run_experiment.sh] Eval failed at step: "$i
    ~/health_check.sh "Eval failed for experiment $TRAIN_DIR at step $i"
    continue
fi
done
~/health_check.sh "Experiment $TRAIN_DIR finished!"


