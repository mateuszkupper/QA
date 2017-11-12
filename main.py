import tensorflow as tf
import numpy as np
import glove as glove  
import re
import string 

glove_dimensionality = 50
largest_num_of_sentences, largest_num_of_words, largest_num_of_words_any_paragraph = glove.count_words_paragraphs_in_squad()
largest_num_of_words_in_answer = glove.get_largest_num_of_words_in_answer()

def get_words(classification, answer_num, paragraph_question_mapping):
	answer_lookup_dict = glove.get_answer_dictionary(answer_num, paragraphs_str, paragraph_question_mapping, largest_num_of_words_any_paragraph+1)
	vectors = [0 for i in range(largest_num_of_words_in_answer)]
	i=0
	for word in classification:
		j=0
		emb_max = 0
		for emb in word:
			if emb > emb_max:
				emb_max = emb
				vector = j	
			j = j + 1
		vectors[i] = vector
		i = i + 1

	ans = ""	
	for vector in vectors:
		for word, word_embedding in answer_lookup_dict.iteritems():
			if word_embedding[vector] == 1:
				try:
					ans = ans + " " + word
				except Exception:
					ans = ans + " " + str(word)
				break
	ans = ans + "."
	return ans[1:].capitalize()

d = 400
global_step = tf.Variable(0, name="global_step")
question = tf.placeholder(tf.float32, shape=(largest_num_of_words, glove_dimensionality), name="question")
text = tf.placeholder(tf.float32, shape=(largest_num_of_sentences, largest_num_of_words, glove_dimensionality), name="text")
answer = tf.placeholder(tf.int32, shape=(largest_num_of_words_in_answer, largest_num_of_words_any_paragraph+1))

A = tf.Variable(tf.random_normal([4, glove_dimensionality, d], stddev=0.1), name="A")
B = tf.Variable(tf.random_normal([4, glove_dimensionality, d], stddev=0.1), name="B")
B1 = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.1), name="B1")
C = tf.Variable(tf.random_normal([4, glove_dimensionality, d], stddev=0.1), name="C")
W = tf.Variable(tf.random_normal([d,largest_num_of_words_any_paragraph+1], stddev=0.1), name="W")
X = tf.Variable(tf.random_normal([largest_num_of_words_in_answer, largest_num_of_words], stddev=0.1), name="X")

text_unstacked = tf.unstack(text)
#key addressing
sentence_question_score = np.zeros((len(text_unstacked)))
index = 0
for sentence in text_unstacked:
	sentence_unstacked = tf.unstack(sentence)
	for word in sentence_unstacked:
		question_unstacked = tf.unstack(question)
		for keyword in question_unstacked:
			if keyword==word:
				sentence_question_score[index] = sentence_question_score[index] + 1
	index = index + 1

#four highest scored sentences
score_sorted = sentence_question_score.argsort()[-4:][::-1]
keys = [0 for i in range(4)]
for a in range(len(keys)):
	keys[a] = text_unstacked[score_sorted[a]]

M = tf.matmul(keys, A)
C_m = tf.matmul(keys, C)		
u = tf.matmul(question, B1)
for i in range(5):
	U = tf.stack([u]*4)
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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)
train = optimizer.minimize(loss)
accuracy = tf.reduce_mean(tf.cast(answer_softmax, tf.float32))
init = tf.global_variables_initializer()

answers, paragraphs, questions, paragraph_question_mapping = glove.vectorise_squad()
a, paragraphs_str, c, paragraph_question_mapping = glove.read_squad()
with tf.Session() as sess:
	init.run()
	for i in range(200):
		answer_num = 0
		for question_x in questions:	
			an = answers[answer_num]
			qu = question_x
			pa = paragraphs[paragraph_question_mapping[answer_num]]
			sess.run(train, feed_dict={question: qu, answer: an, text: pa})
			acc_train = loss.eval(feed_dict={question: qu, answer: an, text: pa})
			print(i ,answer_num, " Train accuracy: ", acc_train)

			feed_dict = {question: qu, text: pa}
			classification = sess.run(answer_softmax, feed_dict)
			print get_words(classification, answer_num, paragraph_question_mapping)
			answer_num = answer_num + 1

	builder = tf.saved_model.builder.SavedModelBuilder("model1")
	builder.add_meta_graph_and_variables(sess, ["tag"], signature_def_map= {
		"model": tf.saved_model.signature_def_utils.predict_signature_def(
		    inputs= {"question": question, "text": text},
		    outputs= {"answer": answer_softmax})
		})
	builder.save()
