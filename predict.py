import tensorflow as tf
import numpy as np
import glove as glove 
import re

glove_dimensionality = 50
largest_num_of_sentences, largest_num_of_words = glove.count_words_paragraphs_in_squad()
d = 20
question = tf.placeholder(tf.float32, shape=(largest_num_of_words, glove_dimensionality))
text = tf.placeholder(tf.float32, shape=(largest_num_of_sentences, largest_num_of_words, glove_dimensionality))
answer = tf.placeholder(tf.int32, shape=(largest_num_of_words, len(glove.glove_lookup)))

A = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=1), name="A")
B = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=1), name="B")
B1 = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=1), name="B1")
C = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=1), name="C")
W = tf.Variable(tf.random_normal([d, len(glove.glove_lookup)], stddev=1), name="W")
saver = tf.train.Saver()
par = "The first inhabitants of North America migrated from Siberia by way of the Bering land bridge and arrived at least 15,000 years ago, though increasing evidence suggests an even earlier arrival. After crossing the land bridge, the first Americans moved southward, either along the Pacific coast or through an interior ice-free corridor between the Cordilleran and Laurentide ice sheets. The Clovis culture appeared around 11,000 BC, and it is considered to be an ancestor of most of the later indigenous cultures of the Americas. While the Clovis culture was thought, throughout the late 20th century, to represent the first human settlement of the Americas, in recent years consensus has changed in recognition of pre-Clovis cultures."

ques = "When did the Clovis culture appear?"
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

M = tf.matmul(text, A)
C = tf.matmul(text, C)
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
PC = tf.matmul(P, C)
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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
training_op = optimizer.minimize(loss)
accuracy = tf.reduce_mean(tf.cast(answer_softmax, tf.float32))
init = tf.global_variables


with tf.Session() as sess:
	saver.restore(sess, "/tmp/model.ckpt")
	feed_dict = {question: questions_words, text: paragraphs_sentences}
	classification = sess.run(answer_softmax, feed_dict)
	vectors = [0 for i in range(largest_num_of_words)]
	i=0
	for word in classification:
		j=0
		emb_max = 0
		for emb in word:
			if emb > emb_max:
				emb_max = emb
				if emb_max == 1:
					print emb_max
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








		
