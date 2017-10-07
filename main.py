import tensorflow as tf
import numpy as np
import glove1 as glove  

glove_dimensionality = 50
largest_num_of_sentences, largest_num_of_words = glove.count_words_paragraphs_in_squad()
print largest_num_of_sentences
d = 50
question = tf.placeholder(tf.float32, shape=(largest_num_of_words, glove_dimensionality))
text = tf.placeholder(tf.float32, shape=(largest_num_of_sentences, largest_num_of_words, glove_dimensionality))
answer = tf.placeholder(tf.int32, shape=(largest_num_of_words, len(glove.glove_lookup)))

A = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.2), name="A")
B = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.2), name="B")
C = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.2), name="C")
W = tf.Variable(tf.random_normal([d, len(glove.glove_lookup)], stddev=0.2), name="W")

paragraph_num = 0

#m = np.zeros(((largest_num_of_sentences, largest_num_of_words, d)))
#c = np.zeros(((largest_num_of_sentences, largest_num_of_words, d)))
i = 0
text_unstacked = tf.unstack(text)#tf.transpose(text, (1, 0, 2))
for sentence in text_unstacked:
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
		
u = tf.matmul(question, B)#largest_num_of_words x d
#p = np.zeros(largest_num_of_words, largest_num_of_words)
o = np.zeros((largest_num_of_words, d))

i=0
for mi in m:#largest_num_of_sentences x largest_num_of_words x d
	#
	print "shape: " + str(mi[0])
	print "shape: " + str(tf.transpose(u))
	#sentence_question_match = mi[0] * tf.transpose(u)
	sentence_question_match = tf.matmul(mi[0], u, transpose_b=True)
	if i == 0:
		p = tf.nn.softmax(sentence_question_match)
	else:
		p = np.vstack((p, sentence_question_match))
	print p[i]
	print c[i]
	i = i + 1
i = 0
for pi, ci in zip(p, c):
	oi = tf.matmul(pi[0], ci[0])
	o = tf.add(oi, o)
	i = i + 1
print len(glove.glove_lookup)
a = tf.add(o, u)
answer_hat = tf.nn.softmax(tf.matmul(a, W))
#logits = fully_connected(answer_hat, 50, activatation_fn=none)
#xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer, logits=answer_hat)
answer_c = tf.cast(answer, tf.float32)
squared_deltas = tf.square(answer_hat - answer_c)
loss = tf.reduce_mean(squared_deltas)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(loss)
#correct = tf.nn.in_top_k(answer_hat, answer, 1)
accuracy = tf.reduce_mean(tf.cast(squared_deltas, tf.float32))
init = tf.global_variables_initializer()
#p 390
paragraphs, questions, answers, paragraph_question_mapping = glove.vectorise_squad()
with tf.Session() as sess:
	init.run()
	answer_num = 0
	for answer_x in answers:
		an = answer_x
		#an = tf.convert_to_tensor(an, dtype=tf.float32)
		print a
		qu = questions[answer_num]
		#qu = tf.convert_to_tensor(qu, dtype=tf.float32)
		pa = paragraphs[paragraph_question_mapping[answer_num]]
		#pa = tf.convert_to_tensor(pa, dtype=tf.float32)
		sess.run(training_op, feed_dict={question: qu, answer: an, text: pa})
		acc_train = accuracy.eval(feed_dict={question: qu, answer: an, text: pa})
		acc_test = accuracy.eval(feed_dict={question: qu, answer: an, text: pa})
		print(answer_num, " T acc: ", acc_train, " Ts acc: ", acc_test)
		answer_num = answer_num + 1
