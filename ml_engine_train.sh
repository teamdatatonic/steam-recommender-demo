JOBNAME=$1     #MF
FULL_JOBNAME=${JOBNAME}_training_$(date +"%Y%m%d_%H%M%S")

gcloud ml-engine jobs submit training $FULL_JOBNAME \
--job-dir gs://example-bucket/models/$MODEL_TYPE/$FULL_JOBNAME \
--region europe-west1 \
--package-path $PWD/trainer \
--module-name trainer.task \
--runtime-version 1.10 \
--python-version 3.5 \
--config hp_config.yaml \
-- \
--MODEL_BUCKET=gs://example-bucket/models/$MODEL_TYPE/$FULL_JOBNAME \
--epochs=20
