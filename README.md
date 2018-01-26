# Soft Spoken
### By Adam Snaider, Mathew Li, Shivan Vipani, and Michael Wono

## About
This project was created at [SB Hacks IV](https://www.sbhacks.com/)

It is a machine learning algorithm to detect the levels of toxicity in online
comments. The idea and the dataset came from [this Kaggle
competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

## Usage
In order to run the training step, one must first clean the data and create the
word embeddings. To do this, run the python script vocab.py. It will generate
all the data needed.

After all the data has been processed, it's time to train the model by running the train.py module.
This model was created using TensorFlow so make sure that TensorFlow is installed.

After train.py finishes its run, you can run the server.py module and navigate
to 'http://127.0.0.1:9876/'. From here, you just write a comment in the text box
and click submit to show the results in the chart
