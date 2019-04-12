import argparse

from google.cloud import storage, bigquery

import data_prep
import gcp

storage_client = storage.Client()
bq_client = bigquery.Client()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default='Games_2')
    parser.add_argument(
        '--gcs_write_bucket',
        type=str,
        default='example-bucket/data/regression/')
    parser.add_argument(
        '--gcs_read_bucket',
        type=str,
        default='example-bucket/data-original-small/')
    parser.add_argument('--local_save_dir', type=str, default='data')

    return parser.parse_args()


def main(args):
    columns_original = ['steamid', 'appid', 'playtime']
    columns_final = ['steamid', 'appid', 'rating']

    # Create BQ dataset for preprocessing
    dataset_ref = bq_client.dataset('Steam')
    if not gcp.dataset_exists(dataset_ref):
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        dataset = bq_client.create_dataset(dataset)

    # Transform data
    data_prep.process_original_steam(
        args.local_save_dir + '/original',
        args.gcs_read_bucket,
        args.gcs_write_bucket,
        args.filename,
        columns_original,
        columns_final)

    # Process data
    data_prep.process_dataset(args.local_save_dir + '/raw', columns_final)
    # Upload training, test and prediction sets to GCS
    bucket_name, prefix = args.gcs_write_bucket.split('/', 1)
    gcp.upload_to_gcs(args.local_save_dir + '/processed', bucket_name,
                            prefix, args.local_save_dir)


if __name__ == '__main__':
    main(args=get_args())
