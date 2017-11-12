import tensorflow as tf
import numpy as np
import glove as glove  
import re
import string 

glove_dimensionality = 50
largest_num_of_sentences, largest_num_of_words, largest_num_of_words_any_paragraph = glove.count_words_paragraphs_in_squad()
largest_num_of_words_in_answer = glove.get_largest_num_of_words_in_answer()
d = 400
global_step = tf.Variable(0, name="global_step")
question = tf.placeholder(tf.float32, shape=(largest_num_of_words, glove_dimensionality), name="question")
text = tf.placeholder(tf.float32, shape=(largest_num_of_sentences, largest_num_of_words, glove_dimensionality), name="text")
answer = tf.placeholder(tf.int32, shape=(largest_num_of_words_in_answer, largest_num_of_words_any_paragraph))

A = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=0.1), name="A")
B = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=0.1), name="B")
B1 = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.1), name="B1")
C = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=0.1), name="C")
W = tf.Variable(tf.random_normal([d,largest_num_of_words_any_paragraph], stddev=0.1), name="W")
X = tf.Variable(tf.random_normal([largest_num_of_words_in_answer, largest_num_of_words], stddev=0.1), name="X")

M = tf.matmul(text, A)
C_m = tf.matmul(text, C)
		
u = tf.matmul(question, B1)
for i in range(2):
	U = tf.stack([u]*largest_num_of_sentences)
	sentence_question_match = tf.matmul(M, U, transpose_b=True)
	P = tf.nn.softmax(sentence_question_match)
	PC = tf.matmul(P, C_m)
	o = tf.reduce_sum(PC, 0)
	u = tf.add(o, u)
a = tf.matmul(X, u)
answer_hat = tf.matmul(a, W)
answer_softmax = tf.nn.softmax(logits=answer_hat, name="answer")
xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=answer_hat, labels=answer)
answer_c = tf.cast(answer, tf.float32)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)
accuracy = tf.reduce_mean(tf.cast(answer_softmax, tf.float32))
init = tf.global_variables_initializer()

answers, paragraphs, questions, paragraph_question_mapping = glove.vectorise_squad()
a, paragraphs_str, c, paragraph_question_mapping = glove.read_squad()
with tf.Session() as sess:
	init.run()
	for i in range(1):
		answer_num = 0
		for question_x in questions:	
			an = answers[answer_num]
			qu = question_x
			pa = paragraphs[paragraph_question_mapping[answer_num]]
			sess.run(train, feed_dict={question: qu, answer: an, text: pa})
			acc_train = loss.eval(feed_dict={question: qu, answer: an, text: pa})
			print(answer_num, " Train accuracy: ", acc_train)

			feed_dict = {question: qu, text: pa}
			classification = sess.run(answer_softmax, feed_dict)
			answer_num = answer_num + 1

	builder = tf.saved_model.builder.SavedModelBuilder("model")
	builder.add_meta_graph_and_variables(sess, ["tag"], signature_def_map= {
		"model": tf.saved_model.signature_def_utils.predict_signature_def(
		    inputs= {"question": question, "text": text},
		    outputs= {"answer": answer_softmax})
		})
	builder.save()
