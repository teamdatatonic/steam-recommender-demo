import argparse
import subprocess

import pandas as pd
import numpy as np
import json
import re
import os
import glob

from google.cloud import storage, bigquery

import queries
import data_prep
import gcp

storage_client = storage.Client()
bq_client = bigquery.Client()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default='Games_2')
    parser.add_argument('--model', type=str, default='MF_GPU')
    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument(
        '--gcs_read_bucket',
        type=str,
        default='example-bucket/data-original-small/')
    parser.add_argument(
        '--gcs_eval_bucket',
        type=str,
        default='example-bucket/evaluation/')
    parser.add_argument('--local_save_dir', type=str, default='data')

    return parser.parse_args()


def main(args):
    print('This job will request predictions for ')

    columns_original = ['steamid', 'appid', 'playtime']

    # Create BQ dataset for preprocessing
    dataset_ref = bq_client.dataset('Steam')
    if not gcp.dataset_exists(dataset_ref):
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        dataset = bq_client.create_dataset(dataset)

    # Upload original data to BQ
    _, _ = data_prep.upload_original_steam(
        args.local_save_dir + '/original',
        args.gcs_read_bucket,
        args.filename,
        columns_original
    )

    query = queries.EVALUATION_SET_SQL.format(model_repo='Steam.played_games')
    df = gcp.run_sql('Steam.played_games', query)
    # save dataframe locally to allow analysis
    df.to_pickle('df.pickle')

    tests_json = df[['steamid', 'appid']].to_json(orient='records')

    eval_instances = 'evaluation/eval_instances.json'
    os.makedirs(os.path.dirname(eval_instances), exist_ok=True)
    with open(eval_instances, 'w') as f:
        f.write(re.sub(r'},{', r'}\n{', tests_json).lstrip('[').rstrip(']'))
        f.flush()
        os.fsync(f.fileno())

    bucket_name, prefix = args.gcs_eval_bucket.split('/', 1)
    gcp.upload_to_gcs(os.path.dirname(eval_instances), bucket_name, prefix, '')

    subprocess.run(
        ['gcloud', 'ml-engine', 'jobs', 'submit', 'prediction', 'evaluate_steam',
         '--data-format', 'text',
         '--input-paths','gs://{}{}'.format(args.gcs_eval_bucket, eval_instances),
         '--output-path','gs://{}{}/{}/{}'.format(args.gcs_eval_bucket, args.model, args.version, 'eval'),
         '--region', 'europe-west1',
         '--model', args.model,
         '--version', args.version,
         '--max-worker-count','1']
    )

    while True:
        x = input(
            'Waiting for predictions to complete. '
            'Check the job (evaluate_steam) and type "done" to proceed.\n'
        )
        if x == 'done':
            break
        else:
            print('Unrecognized command. Type "done" to proceed')


    # Retrieve predictions from GCS
    eval_results = 'evaluation/eval_results.json'
    os.makedirs(os.path.dirname(eval_results), exist_ok=True)
    results_prefix = prefix+'{}/{}/eval'.format(args.model, args.version)
    gcp.download_from_gcs(os.path.dirname(eval_results), bucket_name, results_prefix, match='prediction.results')

    predicted_ratings = []
    for local_file in glob.glob(os.path.dirname(eval_results)+'/prediction.results*'):
        with open(local_file, 'r') as f:
            predicted_ratings.extend([float(json.loads(line)['predictions'][0]) for line in f])

    print(len(predicted_ratings))
    df = pd.read_pickle('df.pickle')
    df = df.assign(predicted=predicted_ratings)
    recommendations = df.groupby('steamid').apply(lambda x: x.nlargest(5, 'predicted'))
    print(recommendations.head(20))


if __name__ == '__main__':
    main(args=get_args())
