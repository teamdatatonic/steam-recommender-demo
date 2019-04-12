""" Training script for factorization of Steam playtime rating matrix.

Usage:
    python3 train.py [options]

Options:
    --version=<str>.       model version name
    --dim=<k>              dimension of the latent space for factorization [default: 3]
    --epochs=<n>           number of epochs to train for [default: 1]
    --lr=<lr>              learning rate while training [default: 0.01]
    --MODEL_BUCKET=<str>   where to save the models on GCS
    --train_data=<str>.    where to access training data in GCS
    --test_data=<str>.     where to access test data in GCS
"""

import tensorflow as tf
import argparse
import sys
import calendar
import time

from trainer.input import input_fn, ml_engine_online_serving_input_receiver_fn
from trainer.model import MF_model_fn

tf.logging.set_verbosity(tf.logging.INFO)

timestamp = str(calendar.timegm(time.gmtime()))


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--version', type=str, default='')

    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument(
        '--MODEL_BUCKET',
        type=str,
        default='gs://example-bucket/models/regression/'
    )
    parser.add_argument(
        '--user_vocab',
        type=str,
        default=
        'gs://example_bucket/data/regression/users.csv',
        help='Path to the user vocab file. ')
    parser.add_argument(
        '--game_vocab',
        type=str,
        default=
        'gs://example_bucket/data/regression/games.csv',
        help='Path to the game vocab file. ')
    parser.add_argument(
        '--train_data',
        type=str,
        default=
        'gs://example-bucket/data/regression/*played_games/train.csv',
        help='Path to the training data. ')
    parser.add_argument(
        '--test_data',
        type=str,
        default=
        'gs://example-bucket/data/regression/*played_games/test.csv',
        help='Path to the test data. ')

    return parser.parse_known_args()


# Create metric for hyperparameter tuning
def my_metrics(labels, predictions):
    return {
        "rmse":
        tf.metrics.root_mean_squared_error(labels, predictions['predictions']),
        "mae":
        tf.metrics.mean_absolute_error(labels, predictions['predictions']),
    }


def main(unused_argv):
    # define feature columns
    users = tf.feature_column.categorical_column_with_vocabulary_file(
        'steamid', FLAGS.user_vocab, dtype=tf.string)
    games = tf.feature_column.categorical_column_with_vocabulary_file(
        'appid', FLAGS.game_vocab, dtype=tf.string)


    # join the hypertuning trial number to the model storage path
    model_checkpoints = FLAGS.MODEL_BUCKET + FLAGS.version + '/model/' + timestamp
    model_serving = FLAGS.MODEL_BUCKET + FLAGS.version + '/serving/' + timestamp
    trial = json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get(
        'trial', '')
    model_checkpoints = os.path.join(model_checkpoints, trial)
    model_serving = os.path.join(model_serving, trial)


    # make matrix factorization estimator
    model_fn = MF_model_fn(users, games, FLAGS.dim, FLAGS.lr)
    run_config = tf.contrib.learn.RunConfig(
        tf_random_seed=42,
        save_checkpoints_secs=60,
        save_summary_steps=1000,
        log_step_count_steps=100
    )
    estimator = tf.estimator.Estimator(
        model_fn, model_dir=model_checkpoints, config=run_config
    )
    estimator = tf.contrib.estimator.add_metrics(estimator, my_metrics)


    # define training procedure
    tf.estimator.train_and_evaluate(
        estimator,
        tf.estimator.TrainSpec(
            input_fn=
            lambda: input_fn(FLAGS.train_data, FLAGS.epochs, shuffle=True)
        ),
        tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(FLAGS.test_data, 1),
            steps=None,
            throttle_secs=60,
            start_delay_secs=120))

    # when finished, export model for serving
    SIRfn = ml_engine_online_serving_input_receiver_fn()
    estimator.export_savedmodel(
        export_dir_base=model_serving, serving_input_receiver_fn=SIRfn)


if __name__ == '__main__':
    FLAGS, unparsed = get_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
