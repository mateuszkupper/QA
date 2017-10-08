import tensorflow as tf
import numpy as np
import glove1 as glove 
import re

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
saver = tf.train.Saver()
par = "We can now implement our regression model. It only takes one line! We multiply the vectorized input images x by the weight matrix W, add the bias b, and compute the softmax probabilities that are assigned to each class."

ques = "What do we multiply?"
print largest_num_of_words
paragraphs_sentences = np.zeros((largest_num_of_sentences, largest_num_of_words, 50))
sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', par)
j = 0
for sentence in sentences:
	words = sentence.split(' ')
	v = 0;
	for word in words:
		characters = list(word)
		if len(characters) > 0:
			if characters[0] in glove.special_chars:
				glove_embedding = glove.get_glove_embedding(characters[0])
				paragraphs_sentences[j][v]=glove_embedding
				v=v+1
				word = word[1:]
			if characters[len(characters)-1] in glove.special_chars:
				word = word[:-1]
			word = word.lower()
			if "'" in word and characters[0] not in "'" and characters[len(characters)-1] not in "'":
				apostrophe_word = word.split("'")
				glove_embedding = glove.get_glove_embedding(apostrophe_word[0])	
				paragraphs_sentences[j][v]=glove_embedding
				v=v+1
				glove_embedding = glove.get_glove_embedding("'" + apostrophe_word[1])	
				paragraphs_sentences[j][v]=glove_embedding
				v=v+1
				print word + " " + apostrophe_word[0] + "'" + apostrophe_word[1]#---------------------------
			else:						
				glove_embedding = glove.get_glove_embedding(word)	
				paragraphs_sentences[j][v]=glove_embedding
				v=v+1				
			if characters[len(characters)-1] in glove.special_chars:
				glove_embedding = glove.get_glove_embedding(characters[len(characters)-1])
				paragraphs_sentences[j][v]=glove_embedding
				v=v+1								
	j=j+1

questions_words = np.zeros((largest_num_of_words, 50))
words = ques.split(' ')
v = 0;
for word in words:
	characters = list(word)
	if len(characters) > 0:
		if characters[0] in glove.special_chars:
			glove_embedding = glove.get_glove_embedding(characters[0])
			questions_words[v]=glove_embedding
			v=v+1
			word = word[1:]
		if characters[len(characters)-1] in glove.special_chars:
			word = word[:-1]
		word = word.lower()
		if "'" in word and characters[0] not in "'" and characters[len(characters)-1] not in "'":
			apostrophe_word = word.split("'")
			glove_embedding = glove.get_glove_embedding(apostrophe_word[0])	
			questions_words[v]=glove_embedding
			v=v+1
			glove_embedding = glove.get_glove_embedding("'" + apostrophe_word[1])	
			questions_words[v]=glove_embedding
			v=v+1
			print word + " " + apostrophe_word[0] + "'" + apostrophe_word[1]#---------------------------
		else:	
			glove_embedding = glove.get_glove_embedding(word)	
			questions_words[v]=glove_embedding
			v=v+1				
		if characters[len(characters)-1] in glove.special_chars:
			glove_embedding = glove.get_glove_embedding(characters[len(characters)-1])
			questions_words[v]=glove_embedding
			v=v+1					

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
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(loss)
#correct = tf.nn.in_top_k(answer_hat, answer, 1)
accuracy = tf.reduce_mean(tf.cast(squared_deltas, tf.float32))
init = tf.global_variables_initializer()


with tf.Session() as sess:
	saver.restore(sess, "/tmp/model.ckpt")
	feed_dict = {question: questions_words, text: paragraphs_sentences}
	classification = sess.run(answer_hat, feed_dict)
	vectors = [0 for i in range(largest_num_of_words)]
	print classification
	i=0
	for word in classification:
		j=0
		emb_max = 0
		for emb in word:
			if emb > emb_max:
				vector = j
			j = j + 1
		vectors[i] = vector
		i = i + 1
	
	for vector in vectors:
		print glove.get_word_from_one_hot_encoded(vector)		
