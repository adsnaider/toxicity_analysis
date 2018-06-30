from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import tensorflow as tf

import vocab
import model

import glog as log

log.setLevel('INFO')

train_tfrecords = 'data/train.tfrecords'
valid_tfrecords = 'data/valid.tfrecords'
test_tfrecords = 'data/test.tfrecords'

learning_rate = 0.01
batch_size = 32
num_hidden = 64
num_epochs = 128

possible_labels = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]


def _parse_function(example_proto):
  features = {
      'text': tf.VarLenFeature(tf.int64),
      'labels': tf.FixedLenFeature([len(possible_labels)], tf.int64)
  }
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features['text'], parsed_features['labels']


checkpoint_dir = 'output/checkpoint/'
summaries_dir = 'output/summaries/'

train_graph = tf.Graph()
with train_graph.as_default():

  train_dataset = tf.data.TFRecordDataset(train_tfrecords)
  train_dataset = train_dataset.map(_parse_function)
  train_dataset = train_dataset.apply(
      tf.contrib.data.batch_and_drop_remainder(batch_size))
  train_dataset = train_dataset.shuffle(buffer_size=1000)
  train_dataset = train_dataset.repeat(num_epochs)

  train_iterator = train_dataset.make_one_shot_iterator()
  train_element, train_label = train_iterator.get_next()

  valid_dataset = tf.data.TFRecordDataset(valid_tfrecords)
  valid_dataset = valid_dataset.map(_parse_function)
  valid_dataset = valid_dataset.shuffle(buffer_size=1000)
  valid_dataset = valid_dataset.apply(
      tf.contrib.data.batch_and_drop_remainder(100))
  valid_dataset = valid_dataset.repeat()

  valid_iterator = valid_dataset.make_one_shot_iterator()
  valid_element, valid_label = valid_iterator.get_next()

  global_step = tf.train.get_or_create_global_step()

  train_model = model.Model(300, vocab.max_features + 2, num_hidden,
                            len(possible_labels))
  logits = train_model.inference(train_element, True, name='train_logits')
  loss = train_model.loss(logits, tf.cast(train_label, tf.float32), name='loss')
  optimizer = train_model.optimizer(
      loss,
      learning_rate=learning_rate,
      global_step=global_step,
      name='GradientDescent')

  with tf.name_scope('predictions'):
    train_predictions = tf.nn.sigmoid(logits, name='train_predictions')
    valid_predictions = tf.nn.sigmoid(
        train_model.inference(valid_element, False),
        name='validation_predictions')

  with tf.name_scope('accuracy'):
    train_accuracy = train_model.accuracy(
        train_predictions, train_label, name='train_accuracy')
    valid_accuracy = train_model.accuracy(
        valid_predictions, valid_label, name='valid_accuracy')

  with tf.name_scope('auc'):
    train_auc, update_train_auc = tf.metrics.auc(train_label, train_predictions)
    valid_auc, update_valid_auc = tf.metrics.auc(valid_label, valid_predictions)

  with tf.name_scope('summaries'):
    loss_summary = tf.summary.scalar('loss', loss)
    train_summary = tf.summary.scalar('train accuracy', train_accuracy)
    validation_summary = tf.summary.scalar('validation accuracy',
                                           valid_accuracy)
    train_auc_summary = tf.summary.scalar('train AUC', train_auc)
    valid_auc_summary = tf.summary.scalar('valid AUC', valid_auc)
    summary_op = tf.summary.merge_all()

  train_op = tf.group(optimizer, update_train_auc, update_valid_auc)

  saver_hook = tf.train.CheckpointSaverHook(
      checkpoint_dir=checkpoint_dir,
      save_secs=60,
      save_steps=None,
      checkpoint_basename='model.ckpt',
      scaffold=None)
  summary_hook = tf.train.SummarySaverHook(
      save_steps=200,
      output_dir=os.path.join(summaries_dir, 'train/'),
      summary_writer=None,
      scaffold=None,
      summary_op=summary_op)

  with tf.train.MonitoredTrainingSession(
      hooks=[saver_hook, summary_hook], checkpoint_dir=checkpoint_dir) as sess:
    log.info('Starting training')
    while not sess.should_stop():
      sess.run(train_op)

test_graph = tf.Graph()
with test_graph.as_default():
  test_dataset = tf.data.TFRecordDataset(test_tfrecords)
  test_dataset = test_dataset.map(_parse_function)
  test_dataset = test_dataset.apply(
      tf.contrib.data.batch_and_drop_remainder(100))

  test_iterator = test_dataset.make_one_shot_iterator()
  test_element, test_label = test_iterator.get_next()

  test_model = model.Model(300, vocab.max_features + 2, num_hidden,
                           len(possible_labels))
  test_predictions = tf.nn.sigmoid(
      test_model.inference(test_element, False), name='test_predictions')
  test_accuracy = test_model.accuracy(
      test_predictions, test_label, name='test_accuracy')
  test_auc, update_op = tf.metrics.auc(test_label, test_predictions)

  test_accuracy_summary = tf.summary.scalar('test accuracy', test_accuracy)
  test_auc_summary = tf.summary.scalar('test AUC', test_auc)
  summary_op = tf.summary.merge_all()
  global_step = tf.train.create_global_step()
  global_step_increment = tf.assign(global_step, global_step + 1)

  test_op = tf.group(update_op, global_step_increment)

  summary_hook = tf.train.SummarySaverHook(
      save_steps=1,
      output_dir=os.path.join(summaries_dir, 'test/'),
      summary_writer=None,
      scaffold=None,
      summary_op=summary_op)

  with tf.train.MonitoredTrainingSession(
      hooks=[summary_hook], checkpoint_dir=checkpoint_dir) as sess:
    log.info('Meassuring testing performance')
    while not sess.should_stop():
      sess.run(test_op)
