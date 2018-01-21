from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

import tensorflow as tf

import sys
import os

from nltk.tokenize import word_tokenize

import pickle
import string


def save(obj, filename):
  print('Saving to {}'.format(filename))
  with open(filename, 'w') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def maybe_load(filename):
  if os.path.isfile(filename):
    with open(filename, 'rb') as f:
      print('Restoring {}'.format(filename))
      return pickle.load(f)
  return None


def get_and_clean_data(data):
  save_file = 'data/cleaned.pickle'
  restore = maybe_load(save_file)
  if (restore is None):
    sentences = data['comment_text'].fillna('CVxTz').values
    printable = set(string.printable)

    print('Cleaning and tokenizing')
    for i in range(len(sentences)):
      if i % 1000 == 0:
        print("Step {}/{}".format(i, len(sentences)))
      sentences[i] = word_tokenize(
          filter(lambda x: x in printable, sentences[i]))
      for x in range(len(sentences[i])):
        sentences[i][x] = sentences[i][x].lower().strip('=\\|/.,?!')

    save(sentences, save_file)
    return sentences
  else:
    return restore


def get_word_count(sentences):
  save_file = 'data/word_count.pickle'
  restore = maybe_load(save_file)
  if (restore is None):
    print('Counting words')
    word_count = {}
    for i in range(len(sentences)):
      if i % 1000 == 0:
        print("Step {}/{}".format(i, len(sentences)))
      for word in sentences[i]:
        if not word in word_count:
          word_count[word] = 0
        word_count[word] += 1

    save(word_count, save_file)
    return word_count
  else:
    return restore


def get_word2index(word_count):
  save_file = 'data/word2index.pickle'
  restore = maybe_load(save_file)
  if (restore is None):
    print('Sorting Dictionary')
    word_count_list = sorted(
        word_count.iteritems(), key=lambda x: x[1], reverse=True)

    word_count_list = word_count_list[:max_features]
    word2index = {}
    i = 1
    for k in word_count_list:
      if i % 1000 == 0:
        print("Step {}/{}".format(i, len(word_count_list)))
      word2index[k[0]] = i
      i += 1

    save(word2index, save_file)
    return word2index
  else:
    return restore


def convert_training_data(word2index, features, dataset):
  save_file = 'data/features.pickle'
  restore = maybe_load(save_file)
  if (restore is None):
    print('Converting words to indeces')
    labels = np.zeros((len(features), len(possible_labels)))
    for i in range(len(features)):
      if i % 1000 == 0:
        print("Step {}/{}".format(i, len(features)))

      for x in range(len(features[i])):
        if features[i][x] in word2index:
          features[i][x] = word2index[features[i][x]]
        else:
          features[i][x] = UNKNOWN

    # Labels
    for k in range(len(possible_labels)):
      labels[:, k] = dataset[possible_labels[k]].values

    save_obj = {'features': features, 'labels': labels}
    save(save_obj, save_file)
    return features, labels
  else:
    return restore['features'], restore['labels']


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def save_to_binary_file(output_filename, text, labels):
  assert (text.shape[0] == labels.shape[0])
  count = 0

  with tf.python_io.TFRecordWriter(output_filename) as writer:
    for i in range(text.shape[0]):
      if i % 1000 == 0:
        print('Processed data: {}/{}'.format(i, text.shape[0]))
        sys.stdout.flush()

      feature = {
          'text': _int64_feature(text[i]),
          'labels': _int64_feature(labels[i])
      }

      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())
      count += 1

  sys.stdout.flush()
  print('Wrote {}/{} datapoints'.format(count, text.shape[0]))


def resize(text, maxlen, empty):
  resized = np.zeros([text.shape[0], maxlen])
  resized.fill(empty)
  for i in range(len(text)):
    if len(text[i]) > maxlen:
      resized[i, :] = text[i][:maxlen]
    else:
      resized[i, :len(text[i])] = text[i][:]
  return resized


max_features = 20000
UNKNOWN = 0
EMPTY = max_features + 1
possible_labels = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]
maxlen = 200

if __name__ == '__main__':
  dataset = pd.read_csv('data/train.csv')
  sentences = get_and_clean_data(dataset)
  word_count = get_word_count(sentences)
  word2index = get_word2index(word_count)
  text, labels = convert_training_data(word2index, sentences, dataset)
  text = resize(text, maxlen, EMPTY).astype(np.int64)
  labels = labels.astype(np.int64)
  print('Text, Labels')
  print(text, labels)
  print(text.shape, labels.shape)

  train_ratio = 0.9
  valid_ratio = 0.05
  test_ratio = 0.05
  size = text.shape[0]

  train_end = int(train_ratio * size)
  valid_end = int(valid_ratio * size) + train_end
  test_end = int(test_ratio * size) + valid_end

  train_text = text[:train_end]
  train_labels = labels[:train_end]
  print(train_text, train_text.shape)
  valid_text = text[train_end:valid_end]
  valid_labels = labels[train_end:valid_end]
  test_text = text[valid_end:test_end]
  test_labels = labels[valid_end:test_end]

  train_file = 'data/train.tfrecords'
  valid_file = 'data/valid.tfrecords'
  test_file = 'data/test.tfrecords'
  if not os.path.isfile(train_file):
    save_to_binary_file(train_file, train_text, train_labels)
  if not os.path.isfile(valid_file):
    save_to_binary_file(valid_file, valid_text, valid_labels)
  if not os.path.isfile(test_file):
    save_to_binary_file(test_file, test_text, test_labels)
