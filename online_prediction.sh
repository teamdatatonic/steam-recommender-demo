# online prediction
# example: bash online_prediction.sh MF_GPU v1 online_instance.json

MODEL_TYPE=$1
VERSION_NAME=$2
TEST_FILE=$3

gcloud ml-engine predict \
--model $MODEL_TYPE \
--version $VERSION_NAME \
--json-instances $TEST_FILE
