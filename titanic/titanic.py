import tensorflow as tf
import pandas as pd
import numpy as np

EPOCH_NUM = 10000
BATCH_SIZE = 64

def next_batch(df, i=None):
    if i is None:
        start = 0
        end = df.shape[0]
    else:
        start = BATCH_SIZE * i
        end = BATCH_SIZE * (i + 1)
    result = df[start:end]

    batch_ys = result[result.columns[0]]
    batch_ys = np.eye(2)[batch_ys]
    batch_xs = result.drop(result.columns[0], axis=1)

    return batch_xs, batch_ys


def split_dataset(df, test_part=None):
    """
    Split dataframe
    :param test_part: float from 0 to 1
    :param df: pandas dataframe
    :return: (pandas dataframe train, pandas dataframe test)
    """
    length = df.shape[0]
    if test_part is None:
        test_part = 0.15

    test_part = int(length * test_part)

    test_dataset = df[0:test_part]
    training_dataset = df[test_part:]
    return training_dataset, test_dataset

data = pd.read_csv('train.csv', usecols=[1, 2, 4, 5, 6, 7, 9, 11], skiprows=[0], header=None)
data.replace(["female", "male"], [0, 1], inplace=True)
data.replace(["Q", "C", "S"], [0, 1, 2], inplace=True)
data = data.dropna()

x = tf.placeholder(tf.float32, [None, 7], name='InputData')
W = tf.Variable(tf.truncated_normal([7, 2], stddev=0.1), name='Weights')
b = tf.Variable(tf.truncated_normal([2], stddev=0.1), name='Bias')
y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer(0.5).minimize(loss)

correct_preds = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

init = tf.global_variables_initializer()

training_dataset, test_narray = split_dataset(data)

with tf.Session() as sess:
    sess.run(init)

    training_dataset_size = data.shape[0]
    for epoch in range(EPOCH_NUM):
        avg_cost = 0.
        total_batch = int(training_dataset_size / BATCH_SIZE)
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(data, i)
            _, c, acc = sess.run([train_step, loss, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
            avg_cost += c / total_batch

        print("Epoch:", '%d' % (epoch + 1), "cost=", "{0}".format(avg_cost), "accuracy=", "{0}".format(acc))

    test_x, test_y = next_batch(test_narray)
    print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

