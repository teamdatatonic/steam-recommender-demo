import tensorflow as tf


def input_fn(path, epochs, shuffle=False):

    # get filenames and shuffle them
    file_list = tf.gfile.Glob(path)
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    if shuffle:
        dataset = dataset.shuffle(50, seed=42)

    # read lines of files as row strings, then shuffle and batch
    f = lambda filepath: tf.data.TextLineDataset(filepath).skip(1)
    dataset = dataset.interleave(f, cycle_length=8, block_length=8)
    if shuffle:
        dataset = dataset.shuffle(100000, seed=42)
    dataset = dataset.batch(1000)

    # parse row strings into features and labels, then return batches
    dataset = dataset.map(parse_csv, num_parallel_calls=4)
    dataset = dataset.cache()
    iterator = dataset.repeat(epochs).make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


def parse_csv(records):
    """Parse csv row strings into feature tensors."""

    steamid_tensor, appid_tensor, playtime_tensor = (tf.decode_csv(
        records, record_defaults=[[''], [''], [0.0]]))

    features = {'steamid': steamid_tensor, 'appid': appid_tensor}
    labels = (playtime_tensor - 2.8)/0.9) # Normalize

    return features, labels


def ml_engine_online_serving_input_receiver_fn():

    features = {
        'steamid': tf.placeholder(shape=[None], dtype=tf.string),
        'appid': tf.placeholder(shape=[None], dtype=tf.string)
    }

    fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features)

    return fn
