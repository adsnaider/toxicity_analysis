from __future__ import division
from __future__ import print_function

import tensorflow as tf
import vocab


def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)


def bias_variable(shape, const=0.0, name=None):
  initial = tf.constant(const, shape=shape)
  return tf.Variable(initial, name=name)


class Model():

  def __init__(self, embedding_size, vocab_size, hidden_size, output_size):
    self.embedding_size = embedding_size
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.embeddings = weight_variable([vocab_size, embedding_size])
    self.w1 = weight_variable([embedding_size, hidden_size], name='W1')
    self.w2 = weight_variable([hidden_size, hidden_size], name='W2')
    self.w3 = weight_variable([hidden_size, hidden_size], name='W3')
    self.w4 = weight_variable([hidden_size, output_size], name='W4')

    self.b1 = bias_variable([hidden_size], name='b1')
    self.b2 = bias_variable([hidden_size], name='b2')
    self.b3 = bias_variable([hidden_size], name='b3')
    self.b4 = bias_variable([output_size], name='b4')

  def inference(self, ids, training, name=None):
    drop = 0.5
    x = tf.reduce_mean(
        tf.nn.embedding_lookup(self.embeddings, ids, name='X'), axis=1)
    hidden = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
    dropout = tf.layers.dropout(hidden, drop, training=training)
    hidden = tf.nn.relu(tf.matmul(dropout, self.w2) + self.b2)
    dropout = tf.layers.dropout(hidden, drop, training=training)
    hidden = tf.nn.relu(tf.matmul(dropout, self.w3) + self.b3)
    dropout = tf.layers.dropout(hidden, drop, training=training)
    return tf.matmul(dropout, self.w4) + self.b4

  def loss(self, logits, labels, name=None):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels),
        name=name)

  def optimizer(self, loss, learning_rate=0.1, global_step=None, name=None):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step)

  def accuracy(self, predictions, labels, name=None):
    return tf.reduce_mean(
        tf.cast(
            tf.equal(tf.cast(tf.round(predictions), tf.int64), labels),
            tf.float32),
        name=name)
