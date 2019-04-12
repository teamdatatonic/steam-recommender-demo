import time
import os
import glob
import re

from google.cloud.exceptions import NotFound
from google.cloud import storage, bigquery
from google.cloud.bigquery import job

storage_client = storage.Client()
bq_client = bigquery.Client()

import data_prep

def dataset_exists(dataset_reference):
    try:
        bq_client.get_dataset(dataset_reference)
        return True
    except NotFound:
        return False


def upload_to_gcs(directory, gcs_bucket, prefix, local_save_dir):
    print('Uploading data to gcs bucket: {}/{}'.format(gcs_bucket, prefix))
    start_time = time.time()
    bucket = storage_client.get_bucket(gcs_bucket)
    assert os.path.isdir(directory)
    for local_file in glob.glob(directory + '/**') + glob.glob(directory +
                                                               '/**/**'):
        if not os.path.isfile(local_file):
            continue

        blob = bucket.blob(
            prefix + local_file.replace(local_save_dir + '/processed/', ''))
        blob.upload_from_filename(local_file)
    print(("* upload_data time: {:.2f}".format(time.time() - start_time)))


def download_from_gcs(directory, gcs_bucket, prefix, match=None):
    print('Downloading data from gcs bucket: ' + gcs_bucket + '/' + prefix)
    start_time = time.time()
    bucket = storage_client.get_bucket(gcs_bucket)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        filepath = blob.name.split('/', prefix.count('/'))[-1]
        dirname = os.path.dirname('{}/{}'.format(directory,
                                                 filepath.split('/', 1)[0]))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        if not os.path.isfile(directory + '/' + filepath):
            pattern1 = re.compile(".*" + match + "(.*?\d){10}")
            pattern2 = re.compile(".*" + match + "(.*?\d){5}-of-(.*?\d){5}")
            if (match is None) or pattern1.match(filepath) or pattern2.match(filepath):
                filepath = filepath[filepath.find(match):]
                blob.download_to_filename(directory + '/' + filepath)
        else:
            print(filepath + ' already exists! Ignoring it..')
    print(("* download_data time: {:.2f}".format(time.time() - start_time)))


def upload_to_bq(filename, schema, table_name):
    client = bigquery.Client()
    dataset_ref = client.dataset('Steam')
    job_config = bigquery.LoadJobConfig()
    job_config.schema = schema
    job_config.skip_leading_rows = 1
    # The source format defaults to CSV, so the line below is optional.
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.write_disposition = job.WriteDisposition.WRITE_TRUNCATE
    with open(filename, 'rb') as source_file:
        load_job = client.load_table_from_file(
            source_file, dataset_ref.table(table_name),
            job_config=job_config)  # API request
    print('Starting job {}'.format(load_job.job_id))

    load_job.result()  # Waits for table load to complete.
    print('Job finished.')

    destination_table = client.get_table(dataset_ref.table(table_name))
    print('Loaded {} rows.'.format(destination_table.num_rows))


def run_sql(model_repo,
            query,
            table_ref=None,
            save_tmp_table=False,
            gcs_write_bucket=None,
            save_directory=None):
    print('Performing requested query from BQ table {}'.format(model_repo))
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig()
    job_config.use_legacy_sql = False
    job_config.write_disposition = job.WriteDisposition.WRITE_TRUNCATE
    if save_tmp_table:
        table_ref = client.dataset('Steam').table(table_ref)
        job_config.destination = table_ref
    query_job = client.query(query=query, job_config=job_config)
    result = query_job.result()
    if save_tmp_table:
        print('Downloading query results from table {}'.format(table_ref.path))
        extract_job = client.extract_table(table_ref,
                                           'gs://' + gcs_write_bucket)
        extract_job.result()
        print('Exported data:{} to {}'.format(model_repo, gcs_write_bucket))
        print('Downloading data from GCS to local...')
        bucket_name, prefix = gcs_write_bucket.split('/', 1)
        prefix, match = prefix.rsplit('/', 1)
        match = match.split('*', 1)[0]
        print(os.path.dirname(save_directory))
        if not os.path.exists(os.path.dirname(save_directory)):
            os.makedirs(os.path.dirname(save_directory))
        download_from_gcs(
            os.path.dirname(save_directory), bucket_name, prefix, match=match)
        df = data_prep.create_pd_dataframe_multiple(
            os.path.dirname(save_directory) + '/' + match + '**', match)
        df.to_csv(save_directory, index=False)
        print('Removing all tmp files')
        for f in os.listdir(os.path.dirname(save_directory)):
            if re.search(match, f):
                print('Removing ' + f)
                os.remove(os.path.join(os.path.dirname(save_directory), f))
    else:
        return result.to_dataframe()
