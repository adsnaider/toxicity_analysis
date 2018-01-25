from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

import os
from flask import Flask, jsonify, request, redirect, url_for
import json

import vocab
import model
import pickle

app = Flask(__name__)
app.config.from_object(__name__)


def predict(message):
  new_message = vocab.clean_sentence([message])[0]

  fle = 'data/word2index.pickle'
  if (os.path.isfile(fle)):
    with open(fle, 'r') as f:
      word2index = pickle.load(f)
  else:
    _, _, _, word2index = vocab.update()

  for i in range(len(new_message)):
    if new_message[i] in word2index:
      new_message[i] = word2index[new_message[i]]
    else:
      new_message[i] = vocab.UNKNOWN

  num_hidden = 64
  possible_labels = [
      'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
  ]

  predict_graph = tf.Graph()
  with predict_graph.as_default():
    tf_input = tf.constant([new_message])
    deep_model = model.Model(300, vocab.max_features + 2, num_hidden,
                             len(possible_labels))
    result = tf.nn.sigmoid(deep_model.inference(tf_input, False, name='out'))
    global_step = tf.train.get_or_create_global_step()

    with tf.train.MonitoredTrainingSession(checkpoint_dir='checkpoint') as sess:
      return sess.run(result)


@app.route('/')
def send_js():
  return app.send_static_file('toxicity.html')


@app.route('/flat-ui.css')
def send_css():
  return app.send_static_file('flat-ui.css')


@app.route('/handle_data', methods=['POST', 'GET'])
def handle_data():
  if request.method == 'POST':
    #  print(request.form)
    msg = request.form['MSG']
    result = predict(msg)
    result = json.dumps(result[0].tolist())
    return result


if __name__ == '__main__':
  app.run(debug=True, port=9876)
