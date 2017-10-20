import tensorflow as tf
import numpy as np
import glove as glove  

glove_dimensionality = 50
largest_num_of_sentences, largest_num_of_words = glove.count_words_paragraphs_in_squad()
print largest_num_of_sentences
d = 1
global_step = tf.Variable(0, name="global_step")
question = tf.placeholder(tf.float32, shape=(largest_num_of_words, glove_dimensionality))
text = tf.placeholder(tf.float32, shape=(largest_num_of_sentences, largest_num_of_words, glove_dimensionality))
answer = tf.placeholder(tf.int32, shape=(largest_num_of_words, len(glove.glove_lookup)))

A = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=1), name="A")
B = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=1), name="B")
B1 = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=1), name="B1")
C = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=1), name="C")
W = tf.Variable(tf.random_normal([d, len(glove.glove_lookup)], stddev=1), name="W")

M = tf.matmul(text, A)
C_m = tf.matmul(text, C)
print M.get_shape()
		

u = tf.matmul(question, B1)
question_stacked = tf.stack([question]*largest_num_of_sentences)
print question_stacked.get_shape()
U = tf.matmul(question_stacked, B)
print U.get_shape()
sentence_question_match = tf.matmul(M, U, transpose_b=True)
print sentence_question_match.get_shape()
P = tf.nn.softmax(sentence_question_match)
print P.get_shape()
PC = tf.matmul(P, C_m)
print PC.get_shape()
o = tf.reduce_sum(PC, 0)
print o.get_shape()
a = tf.add(o, u)
print a.get_shape()
answer_hat = tf.matmul(a, W)
print answer_hat.get_shape()
answer_softmax = tf.nn.softmax(logits=answer_hat)
xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=answer_hat, labels=answer)
print answer_softmax.get_shape()
answer_c = tf.cast(answer, tf.float32)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)


params = [A, B, C, W, B1]
grads_and_vars = optimizer.compute_gradients(xentropy, params)
clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], 50), gv[1]) \
                           for gv in grads_and_vars]

inc = global_step.assign_add(1)
with tf.control_dependencies([inc]):
	optim = optimizer.apply_gradients(clipped_grads_and_vars)



#training_op = optimizer.minimize(loss)
accuracy = tf.reduce_mean(tf.cast(answer_softmax, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

paragraphs, questions, paragraph_question_mapping = glove.vectorise_squad()
a, b, c, paragraph_question_mapping = glove.read_squad()
with tf.Session() as sess:
	init.run()
	answer_num = 0
	for question_x in questions:
		answers_words = np.zeros((largest_num_of_words, len(glove.glove_lookup)))
		answerc = c[answer_num]
		words = answerc.split(' ')
		v = 0;
		for word in words:
			characters = list(word)
			if len(characters) > 0:
				if characters[0] in glove.special_chars:
					glove_embedding = glove.get_one_hot_encoded_from_glove(characters[0])
					answers_words[v]=glove_embedding
					v=v+1
					word = word[1:]
				if characters[len(characters)-1] in glove.special_chars:
					word = word[:-1]
				word = word.lower()
				if "'" in word and characters[0] not in "'" and characters[len(characters)-1] not in "'":
					apostrophe_word = word.split("'")
					glove_embedding = glove.get_one_hot_encoded_from_glove(apostrophe_word[0])	
					answers_words[v]=glove_embedding
					v=v+1
					glove_embedding = glove.get_one_hot_encoded_from_glove("'" + apostrophe_word[1])	
					answers_words[v]=glove_embedding
					v=v+1
				else:	
					glove_embedding = glove.get_one_hot_encoded_from_glove(word)	
					answers_words[v]=glove_embedding
					v=v+1				
				if characters[len(characters)-1] in glove.special_chars:
					glove_embedding = glove.get_one_hot_encoded_from_glove(characters[len(characters)-1])
					answers_words[v]=glove_embedding
					v=v+1					

		an = answers_words
		qu = question_x
		pa = paragraphs[paragraph_question_mapping[answer_num]]
		print pa.shape
		sess.run(answer_hat, feed_dict={question: qu, answer: an, text: pa})
		acc_train = accuracy.eval(feed_dict={question: qu, answer: an, text: pa})
		print(answer_num, " Train accuracy: ", acc_train)
		answer_num = answer_num + 1
	save_path = saver.save(sess, "/tmp/model.ckpt")

