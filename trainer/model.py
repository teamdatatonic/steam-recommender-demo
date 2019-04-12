import tensorflow as tf


def MF_model_fn(users, items, dim, learning_rate, n_classes=6):

    # one-hot encoding
    users_1hot = tf.feature_column.indicator_column(users)
    items_1hot = tf.feature_column.indicator_column(items)

    # scalar embedding (bias terms)
    user_bias_emb = tf.feature_column.embedding_column(users, 1)
    item_bias_emb = tf.feature_column.embedding_column(items, 1)

    def model_fn(features, labels, mode, config):

        # pass features through columns to get input tensors
        users = tf.feature_column.input_layer(features, users_1hot)
        items = tf.feature_column.input_layer(features, items_1hot)

        user_bias = tf.feature_column.input_layer(features, user_bias_emb)
        item_bias = tf.feature_column.input_layer(features, item_bias_emb)

        # weight matrix for factor embedding
        V_u = tf.Variable(tf.random_normal([users.get_shape().as_list()[1], dim], stddev=0.01))
        V_i = tf.Variable(tf.random_normal([items.get_shape().as_list()[1], dim], stddev=0.01))

        # second-order embedding product
        interactions = 0.5*tf.reduce_sum(
            (tf.matmul(users, V_u) + tf.matmul(items, V_i))**2 -
            (tf.matmul(users, V_u**2) + tf.matmul(items, V_i**2)),
            axis=1,
            keep_dims=True
        )

        # scalar for overall bias
        global_bias = tf.Variable(tf.random_normal([], stddev=0.01))

        output = interactions + user_bias + item_bias + global_bias

        # define loss and training operations
        head = tf.contrib.estimator.regression_head()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        spec = head.create_estimator_spec(
             features=features,
             mode=mode,
             logits=output,
             labels=labels,
             optimizer=optimizer)
        return spec

    return model_fn
