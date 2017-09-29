import tensorflow as tf
import numpy as np
from glove  

glove_dimensionality = 50
largest_num_of_sentences, largest_num_of_words = glove.count_words_paragraphs_in_squad()

d = 50
question = tf.placeholder(tf.float32, shape=(largest_num_of_words, glove_dimensionality))
sentence = tf.placeholder(tf.float32, shape=(largest_num_of_words, glove_dimensionality))
answer = tf.placeholder(tf.float32, shape=(largest_num_of_words, glove_dimensionality))

A = tf.Variable(tf.random_normal([glove_dimensionality, d]), stdev=0.2, name="A")
B = tf.Variable(tf.random_normal([glove_dimensionality, d]), stdev=0.2, name="B")
C = tf.Variable(tf.random_normal([glove_dimensionality, d]), stdev=0.2, name="C")

m = tf.matmul(sentence, A)
c = tf.matmul(sentence, C)
u = tf.matmul(question, B)

sentence_question_match = tf.matmul(m, u, transpose_b=True)

p = tf.softmax(sentence_question_match)
oi = tf.matmul(p, c)

W = tf.Variable(tf.random_normal([d, glove_dimensionality]), stdev=0.2, name="W")

o = tf.placeholder(tf.float32, shape=(largest_number_of_sentences, d))

a = softmax(tf.matmul(o+u, W))
