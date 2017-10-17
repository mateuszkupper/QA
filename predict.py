import tensorflow as tf
import numpy as np
import glove as glove 
import re

glove_dimensionality = 50
largest_num_of_sentences, largest_num_of_words = glove.count_words_paragraphs_in_squad()
d = 5
question = tf.placeholder(tf.float32, shape=(largest_num_of_words, glove_dimensionality))
text = tf.placeholder(tf.float32, shape=(largest_num_of_sentences, largest_num_of_words, glove_dimensionality))
answer = tf.placeholder(tf.int32, shape=(largest_num_of_words, len(glove.glove_lookup)))

A = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.5), name="A")
B = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.5), name="B")
C = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.5), name="C")
W = tf.Variable(tf.random_normal([d, len(glove.glove_lookup)], stddev=0.5), name="W")
saver = tf.train.Saver()
par = "The first inhabitants of North America migrated from Siberia by way of the Bering land bridge and arrived at least 15,000 years ago, though increasing evidence suggests an even earlier arrival. [53] After crossing the land bridge, the first Americans moved southward, either along the Pacific coast [54][55] or through an interior ice-free corridor between the Cordilleran and Laurentide ice sheets. [56] The Clovis culture appeared around 11,000 BC, and it is considered to be an ancestor of most of the later indigenous cultures of the Americas. [57] While the Clovis culture was thought, throughout the late 20th century, to represent the first human settlement of the Americas, [58] in recent years consensus has changed in recognition of pre-Clovis cultures. [59]"

ques = "What is the text about?"
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
		else:	
			glove_embedding = glove.get_glove_embedding(word)	
			questions_words[v]=glove_embedding
			v=v+1				
		if characters[len(characters)-1] in glove.special_chars:
			glove_embedding = glove.get_glove_embedding(characters[len(characters)-1])
			questions_words[v]=glove_embedding
			v=v+1					

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


with tf.Session() as sess:
	saver.restore(sess, "/tmp/model.ckpt")
	feed_dict = {question: questions_words, text: paragraphs_sentences}
	classification = sess.run(answer_hat, feed_dict)
	vectors = [0 for i in range(largest_num_of_words)]
	i=0
	for word in classification:
		j=0
		emb_max = 0
		for emb in word:
			if emb > emb_max:
				emb_max = emb
				if emb_max > 0.8:
					vector = j
				else:
					vector = -1	
			j = j + 1
		vectors[i] = vector
		i = i + 1
	
	print " "
	print "Answer: "
	answer = ""	
	for vector in vectors:
		if vector != -1:
			answer = answer + " " + glove.get_word_from_one_hot_encoded(vector)
	answer = answer + "."
	print answer[1:].capitalize()		








		
