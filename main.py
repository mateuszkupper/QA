import tensorflow as tf
import numpy as np
import glove  

glove_dimensionality = 50
largest_num_of_sentences, largest_num_of_words = glove.count_words_paragraphs_in_squad()
print largest_num_of_sentences
d = 50
question = tf.placeholder(tf.float32, shape=(largest_num_of_words, glove_dimensionality))
text = tf.placeholder(tf.float32, shape=(largest_num_of_sentences, largest_num_of_words, glove_dimensionality))
answer = tf.placeholder(tf.float32, shape=(largest_num_of_words, glove_dimensionality))

A = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.2), name="A")
B = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.2), name="B")
C = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.2), name="C")
W = tf.Variable(tf.random_normal([d, d], stddev=0.2), name="W")

paragraph_num = 0

#m = np.zeros(((largest_num_of_sentences, largest_num_of_words, d)))
#c = np.zeros(((largest_num_of_sentences, largest_num_of_words, d)))
i = 0
text = tf.unstack(text)#tf.transpose(text, (1, 0, 2))
for sentence in text:
	if i == 0:
		m = tf.matmul(sentence, A)
		c = tf.matmul(sentence, A)
		print sentence
		print A
	else:
		print sentence.shape
		print tf.matmul(sentence, A).shape
		print m.shape
		m = np.vstack((m, tf.matmul(sentence, A)))
		c = np.vstack((c, tf.matmul(sentence, A)))
	#m[i, :, :] = tf.matmul(sentence, A)
	#c[i, :, :] = tf.matmul(sentence, C)
	i = i + 1
		
u = tf.matmul(question, B)
#p = np.zeros(largest_num_of_words, largest_num_of_words)
o = np.zeros((largest_num_of_words, d))
i=0
for mi in m:
	#sentence_question_match = tf.matmul(mi, u, transpose_b=True)
	sentence_question_match = mi * tf.transpose(u)
	if i == 0:
		p = tf.nn.softmax(sentence_question_match)
	else:
		p = np.vstack((p, sentence_question_match))
	print p[i].shape
	print c[i].shape
	i = i + 1
i = 0
for pi in p:
	oi = pi + c[i]
	o = tf.add(oi, o)
	i = i + 1

a = tf.add(o, u)
answer_hat = tf.nn.softmax(tf.matmul(a, W))
logits = fully_connected(answer_hat, 50, activatation_fn=none)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, answer, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
#p 390
paragraphs, questions, answers, paragraph_question_mapping = glove.vectorise_squad()
with tf.Session() as sess:
	init.run()
	answer_num = 0
	for answer_x in answers:
		a = answer_x
		q = questions[answer_num]
		p = paragraphs[paragraph_question_mapping[answer_num]]
		sess.run(training_op, feed_dict={question: q, answer: a, text: p})
		acc_train = accuracy.eval(feed_dict={question: q, answer: a, text: p})
		acc_test = accuracy.eval(feed_dict={question: q, answer: a, text: p})
		print(answer_num, " T acc: ", acc_train, " Ts acc: ", acc_test)
		anwer_num = anwer_num + 1
