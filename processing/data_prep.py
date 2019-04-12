import time
import csv
import os
import glob

import pandas as pd
import numpy as np

from google.cloud import storage, bigquery
from sklearn.model_selection import train_test_split

import queries
import gcp

storage_client = storage.Client()
bq_client = bigquery.Client()


def create_pd_dataframe_multiple(files, filename, match=None):
    # Read dataset
    files = glob.glob(files)
    list_ = []
    print('Reading ' + filename + ' files:')
    for file_ in files:
        if match is None or match in file_:
            print(file_)
            df_temp = pd.read_csv(file_, index_col=None, header=0)
            list_.append(df_temp)
    print('Files read.')
    df = pd.concat(list_, axis=0, ignore_index=True)
    return df


def rating(x, percentiles):
    if x < percentiles['pctl_25']:
        return 1
    elif x < percentiles['pctl_50']:
        return 2
    elif x < percentiles['pctl_75']:
        return 3
    elif x < percentiles['pctl_95']:
        return 4
    return 5


def upload_original_steam(directory,
                          gcs_read_bucket,
                          filename,
                          columns_original):
    bucket_name, prefix = gcs_read_bucket.split('/', 1)
    gcp.download_from_gcs(directory, bucket_name, prefix, match=filename)
    df = create_pd_dataframe_multiple(directory + '/**', 'original steam',
                                      filename)
    print('Creating played interactions...')
    df.rename(columns={'playtime_forever': 'playtime'}, inplace=True)
    df = df[columns_original]

    # Save true interactions locally
    new_directory = directory.replace('original', 'raw')
    if not os.path.exists(new_directory + '/played_games'):
        os.makedirs(new_directory + '/played_games/')
    if not os.path.exists(new_directory + '/unplayed_games'):
        os.makedirs(new_directory + '/unplayed_games/')
    df.to_csv(new_directory + '/played_games/original.csv', index=False)

    # Upload played games to BQ
    schema = [
        bigquery.SchemaField('steamid', 'STRING'),
        bigquery.SchemaField('appid', 'STRING'),
        bigquery.SchemaField('playtime', 'FLOAT')
    ]
    gcp.upload_to_bq(new_directory + '/played_games/original.csv',
                           schema, 'played_games')

    return df, new_directory


def process_original_steam(directory,
                           gcs_read_bucket,
                           gcs_write_bucket,
                           filename,
                           columns_original,
                           columns_final):

    df, new_directory = upload_original_steam(
        directory,
        gcs_read_bucket,
        filename,
        columns_original
    )

    # Define rating from 0-5 from percentiles of playtime
    print('Obtaining rating from playtime')
    query = queries.PERCENTILE_SQL.format(model_repo='Steam.played_games')
    percentiles = gcp.run_sql('Steam.played_games', query)
    df['rating'] = df['playtime'].map(
        lambda x: rating(x, percentiles.iloc[0].to_dict()))
    df = df[columns_final]
    df.to_csv(new_directory + '/played_games/original.csv', index=False)

    # Create and save fake interactions
    print('Creating unplayed interactions...')
    unique_users = df['steamid'].drop_duplicates().to_frame()
    unique_games = df['appid'].drop_duplicates().to_frame()
    print('Uploading user and games to BQ...')
    unique_users.to_gbq(
        'Steam.users',
        'example-project',
        table_schema=[{
            'name': 'steamid',
            'type': 'STRING'
        }],
        if_exists='replace')
    unique_games.to_gbq(
        'Steam.games',
        'example-project',
        table_schema=[{
            'name': 'appid',
            'type': 'STRING'
        }],
        if_exists='replace')
    unique_users.to_csv(new_directory + '/users.csv', index=False)
    unique_games.to_csv(new_directory + '/games.csv', index=False)


    print('Running query to define fake interactions...')
    query = queries.FAKE_INTERACTIONS_SQL.format(
        model_repo='Steam.played_games', label=columns_final[-1])
    df_unplayed = gcp.run_sql(
        'Steam.played_games',
        query,
        table_ref='fake_interactions',
        save_tmp_table=True,
        gcs_write_bucket=gcs_write_bucket + 'raw/fake_interactions*.csv',
        save_directory=new_directory + '/unplayed_games/original.csv'
    )

    # Delete BQ datasets
    dataset_ref = bq_client.dataset('Steam')
    bq_client.delete_dataset(dataset_ref, delete_contents=True)
    print('Dataset Steam deleted.')


def process_dataset(directory, columns):
    print('Processing data from directory: ' + directory)
    start_time = time.time()

    # Read dataset
    played_games_files = '{}/played_games/*.csv'.format(directory)
    unplayed_games_files = '{}/unplayed_games/*.csv'.format(directory)
    df_played = create_pd_dataframe_multiple(played_games_files, 'raw played')
    df_unplayed = create_pd_dataframe_multiple(unplayed_games_files,
                                               'raw unplayed')

    print('Processing starting...')

    # Balance negative and positive interactions to be the same size
    print('Balancing datasets...')
    drop_indices = np.random.choice(
        df_unplayed.index, len(df_unplayed) - len(df_played), replace=False)
    df_unplayed = df_unplayed.drop(drop_indices)

    print('df_played: ' + str(df_played.shape))
    print('df_unplayed: ' + str(df_unplayed.shape))

    # Split train and test sets
    print('Creating training and test set...')
    train_played, test_played = train_test_split(df_played, test_size=0.2)
    train_unplayed, test_unplayed = train_test_split(
        df_unplayed, test_size=0.2)

    # Save datasets locally
    new_directory = directory.replace('raw', 'processed')

    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
        os.makedirs(new_directory + '/played_games/')
        os.makedirs(new_directory + '/unplayed_games/')

    train_played.to_csv(new_directory + '/played_games/train.csv', index=False)
    train_unplayed.to_csv(
        new_directory + '/unplayed_games/train.csv', index=False)
    test_played.to_csv(new_directory + '/played_games/test.csv', index=False)
    test_unplayed.to_csv(
        new_directory + '/unplayed_games/test.csv', index=False)

    # Create user and game vocabularies
    print('Creating users and games vocabularies...')
    users = pd.concat([df_played['steamid'], df_unplayed['steamid']],
                      axis=0,
                      ignore_index=True).unique()

    games = pd.concat([df_played['appid'], df_unplayed['appid']],
                      axis=0,
                      ignore_index=True).unique()

    cw_users = csv.writer(open(new_directory + '/users.csv', 'w'))
    for user in list(users):
        cw_users.writerow([user])

    cw_games = csv.writer(open(new_directory + '/games.csv', 'w'))
    for game in list(games):
        cw_games.writerow([game])

    print(("* Processing_data time: {:.2f}".format(time.time() - start_time)))
