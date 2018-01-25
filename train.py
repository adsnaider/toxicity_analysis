from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

import vocab
import model

train_tfrecords = 'data/train.tfrecords'
valid_tfrecords = 'data/valid.tfrecords'
test_tfrecords = 'data/test.tfrecords'

learning_rate = 0.1
batch_size = 32
num_hidden = 64
num_epochs = 16

possible_labels = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]

train_graph = tf.Graph()
with train_graph.as_default():

  def _parse_function(example_proto):
    features = {
        'text': tf.FixedLenFeature([vocab.maxlen], tf.int64),
        'labels': tf.FixedLenFeature([len(possible_labels)], tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features['text'], parsed_features['labels']

  train_dataset = tf.data.TFRecordDataset(train_tfrecords)
  train_dataset = train_dataset.map(_parse_function)
  train_dataset = train_dataset.apply(
      tf.contrib.data.batch_and_drop_remainder(batch_size))
  train_dataset = train_dataset.shuffle(buffer_size=1000)
  train_dataset = train_dataset.repeat(num_epochs)

  train_iterator = train_dataset.make_one_shot_iterator()
  train_element, train_label = train_iterator.get_next()
  print(train_dataset)

  valid_dataset = tf.data.TFRecordDataset(valid_tfrecords)
  valid_dataset = valid_dataset.map(_parse_function)
  valid_dataset = valid_dataset.shuffle(buffer_size=1000)
  valid_dataset = valid_dataset.apply(
      tf.contrib.data.batch_and_drop_remainder(100))
  valid_dataset = valid_dataset.repeat()

  valid_iterator = valid_dataset.make_one_shot_iterator()
  valid_element, valid_label = valid_iterator.get_next()
  """
  test_dataset = tf.data.TFRecordDataset(test_tfrecords)
  test_dataset = test_dataset.map(_parse_function)
  test_dataset = test_dataset.repeat()

  test_iterator = test_dataset.make_one_shot_iterator()
  test_element, test_label = test_iterator.get_next()
  print(test_element)
  """

  global_step = tf.train.get_or_create_global_step()

  model = model.Model(300, vocab.max_features + 2, num_hidden,
                      len(possible_labels))
  logits = model.inference(train_element, True, name='train_logits')
  loss = model.loss(logits, tf.cast(train_label, tf.float32), name='loss')
  optimizer = model.optimizer(
      loss, learning_rate=learning_rate, global_step=global_step)

  with tf.name_scope('predictions'):
    train_predictions = tf.nn.sigmoid(logits, name='train-predictions')
    valid_predictions = tf.nn.sigmoid(
        model.inference(valid_element, False), name='validation-predictions')
    #  test_predictions = tf.nn.sigmoid(
    #  model.inference(test_element, False), name='test-predictions')
  with tf.name_scope('accuracy'):
    train_accuracy = model.accuracy(
        train_predictions, train_label, name='train-accuracy')
    valid_accuracy = model.accuracy(
        valid_predictions, valid_label, name='valid-accuracy')
    #  test_accuracy = model.accuracy(
    #  test_predictions, test_label, name='test-accuracy')

  with tf.name_scope('summaries'):
    loss_summary = tf.summary.scalar('loss', loss)
    train_summary = tf.summary.scalar('train accuracy', train_accuracy)
    validation_summary = tf.summary.scalar('validation accuracy',
                                           valid_accuracy)
    #  test_summary = tf.summary.scalar('test accuracy', test_accuracy)

    train_step_summary = tf.summary.merge(
        inputs=[loss_summary, train_summary, validation_summary])

  saver_hook = tf.train.CheckpointSaverHook(
      checkpoint_dir='checkpoint/',
      save_secs=300,
      save_steps=None,
      saver=tf.train.Saver(),
      checkpoint_basename='model.ckpt',
      scaffold=None)
  summary_hook = tf.train.SummarySaverHook(
      save_steps=None,
      save_secs=60,
      output_dir='summaries/',
      summary_writer=None,
      scaffold=None,
      summary_op=train_step_summary)

  with tf.train.MonitoredTrainingSession(
      hooks=[saver_hook, summary_hook], checkpoint_dir='checkpoint/') as sess:
    while not sess.should_stop():
      sess.run(optimizer)

  #  with tf.train.MonitoredTrainingSession(checkpoint_dir='checkpoint/') as sess:
  #  testing_accuracy = sess.run(test_accuracy)
  #  print('Finished training with a testing accuracy of {}\%'.format(
  #  testing_accuracy * 100))
