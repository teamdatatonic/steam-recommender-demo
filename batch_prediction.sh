# use a deployed batch serving model on ML Engine to get predictions
# example: bash batch_prediction.sh MF_GPU v1 {BUCKET}

MODEL_TYPE=$1
VERSION_NAME=$2
BUCKET=$3

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JOB_NAME=${MODEL_TYPE}_predictions_$TIMESTAMP
MAX_WORKER_COUNT="1"

gcloud ml-engine jobs submit prediction $JOB_NAME \
	--data-format=text \
	--input-paths=gs://${BUCKET}/test-predictions/batch_instances.json \
	--output-path=gs://${BUCKET}/test-predictions/${MODEL_TYPE}/${VERSION_NAME}/${JOB_NAME} \
	--region=europe-west1 \
	--model=${MODEL_TYPE} \
	--version=${VERSION_NAME} \
	--max-worker-count=${MAX_WORKER_COUNT}
