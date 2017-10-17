import tensorflow as tf
import numpy as np
import glove as glove  

glove_dimensionality = 50
largest_num_of_sentences, largest_num_of_words = glove.count_words_paragraphs_in_squad()
print largest_num_of_sentences
d = 5
question = tf.placeholder(tf.float32, shape=(largest_num_of_words, glove_dimensionality))
text = tf.placeholder(tf.float32, shape=(largest_num_of_sentences, largest_num_of_words, glove_dimensionality))
answer = tf.placeholder(tf.int32, shape=(largest_num_of_words, len(glove.glove_lookup)))

A = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.5), name="A")
B = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.5), name="B")
C = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.5), name="C")
W = tf.Variable(tf.random_normal([d, len(glove.glove_lookup)], stddev=0.5), name="W")

paragraph_num = 0
i = 0
text_unstacked = tf.unstack(text)
for sentence in text_unstacked:
	if i == 0:
		m = tf.matmul(sentence, A)
		c = tf.matmul(sentence, A)
	else:
		m = np.vstack((m, tf.matmul(sentence, A)))
		c = np.vstack((c, tf.matmul(sentence, A)))
	i = i + 1
		
u = tf.matmul(question, B)
o = np.zeros((largest_num_of_words, d))

i=0
for mi in m:
	sentence_question_match = tf.matmul(mi[0], u, transpose_b=True)
	if i == 0:
		p = tf.nn.softmax(sentence_question_match)
	else:
		p = np.vstack((p, sentence_question_match))
	i = i + 1
i = 0
for pi, ci in zip(p, c):
	oi = tf.matmul(pi[0], ci[0])
	o = tf.add(oi, o)
	i = i + 1

a = tf.add(o, u)
answer_hat = tf.nn.softmax(tf.matmul(a, W))
xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=answer, logits=answer_hat)
answer_c = tf.cast(answer, tf.float32)
squared_deltas = tf.square(answer_hat - answer_c)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
training_op = optimizer.minimize(loss)
accuracy = tf.reduce_mean(tf.cast(xentropy, tf.float32))
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
		sess.run(answer_hat, feed_dict={question: qu, answer: an, text: pa})
		acc_train = accuracy.eval(feed_dict={question: qu, answer: an, text: pa})
		print(answer_num, " Train accuracy: ", acc_train)
		answer_num = answer_num + 1
	save_path = saver.save(sess, "/tmp/model.ckpt")

