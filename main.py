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

with tf.Session() as sess:
	paragraphs, questions, answers, paragraph_question_mapping = glove.vectorise_squad()
	paragraph_num = 0
	for paragraph in paragraphs:
		m = np.zeros(len(paragraph))
		c = np.zeros(len(paragraph))
		i = 0
		for sentence in paragraph:
			m[i] = tf.matmul(sentence, A)
			c[i] = tf.matmul(sentence, C)
			i = i + 1
		u = np.zeros(len(questions))
		i=0
		for question in questions:
			if paragraph_question_mapping[i]==paragraph_num:			
				q[i] = tf.matmul(question, B)
				p = np.zeros(largest_num_of_words, largest_num_of_words)
				j = 0
				o = np.zeros(largest_num_of_words, largest_num_of_words)
				for mi in m:
					sentence_question_match = tf.matmul(mi[j], u[j], transpose_b=True)
					p[j] = tf.softmax(sentence_question_match)
					oi = tf.add(p[j], c[j])
					o = tf.add(oi, o)
				
			i = i + 1
		
		paragraph_num = paragraph_num + 1
		
				
	m = tf.matmul(sentence, A)
	c = tf.matmul(sentence, C)
	u = tf.matmul(question, B)

	sentence_question_match = tf.matmul(m, u, transpose_b=True)

	p = tf.softmax(sentence_question_match)
	oi = tf.matmul(p, c)

	W = tf.Variable(tf.random_normal([d, glove_dimensionality]), stdev=0.2, name="W")

	o = tf.placeholder(tf.float32, shape=(largest_number_of_sentences, d))

	a = softmax(tf.matmul(o+u, W))
