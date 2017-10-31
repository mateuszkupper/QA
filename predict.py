import tensorflow as tf
import numpy as np
import glove as glove  
import re
glove_dimensionality = 50
largest_num_of_sentences, largest_num_of_words = glove.count_words_paragraphs_in_squad()
largest_num_of_words_in_answer = glove.get_largest_num_of_words_in_answer()
print largest_num_of_sentences
d = 10
global_step = tf.Variable(0, name="global_step")
question = tf.placeholder(tf.float32, shape=(largest_num_of_words, glove_dimensionality))
text = tf.placeholder(tf.float32, shape=(largest_num_of_sentences, largest_num_of_words, glove_dimensionality))
answer = tf.placeholder(tf.int32, shape=(largest_num_of_words_in_answer, len(glove.glove_lookup)))

A = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=1), name="A")
B = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=1), name="B")
B1 = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=1), name="B1")
C = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=1), name="C")
W = tf.Variable(tf.random_normal([d, len(glove.glove_lookup)], stddev=1), name="W")
X = tf.Variable(tf.random_normal([largest_num_of_words_in_answer, largest_num_of_words], stddev=1), name="X")
saver = tf.train.Saver()
par = "I was hired at my current workplace just over a month ago. During the interview process, I found out that someone I knew from university was also interviewing for the position. I'll call him Jim. Jim and I have never been close, but are merely aware of each other. Based solely on technical knowledge and relevant expertise, I believe that Jim would have been the better candidate for the job. In particular, he previously worked at a well known company in a position very similar to this one. I was therefore somewhat surprised when I received the offer."

ques = "Where were you hired just over month ago?"
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
C_m = tf.matmul(text, C)
print M.get_shape()
		
#a = tf.Variable(tf.random_normal([largest_num_of_words, d], stddev=1), name="W")
u = tf.matmul(question, B1)
#u = s + a
for i in range(2):
	#print question_stacked.get_shape()
	#U = tf.matmul(question_stacked, B)
	U = tf.stack([u]*largest_num_of_sentences)
	print U.get_shape()
	sentence_question_match = tf.matmul(M, U, transpose_b=True)
	print sentence_question_match.get_shape()
	P = tf.nn.softmax(sentence_question_match)
	print P.get_shape()
	PC = tf.matmul(P, C_m)
	print PC.get_shape()
	o = tf.reduce_sum(PC, 0)
	print o.get_shape()
	u = tf.add(o, u)
a = tf.matmul(X, u)
print a.get_shape()
answer_hat = tf.matmul(a, W)
print answer_hat.get_shape()
answer_softmax = tf.nn.softmax(logits=answer_hat)
xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=answer_hat, labels=answer)
print answer_softmax.get_shape()
answer_c = tf.cast(answer, tf.float32)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)


#params = [A, B, C, W, B1]
#grads_and_vars = optimizer.compute_gradients(xentropy, params)
#clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], 50), gv[1]) \
#                           for gv in grads_and_vars]

#inc = global_step.assign_add(1)
#with tf.control_dependencies([inc]):
#	optim = optimizer.apply_gradients(clipped_grads_and_vars)



#training_op = optimizer.minimize(loss)
accuracy = tf.reduce_mean(tf.cast(answer_softmax, tf.float32) - tf.cast(answer, tf.float32))
init = tf.global_variables


with tf.Session() as sess:
	saver.restore(sess, "/tmp/model.ckpt")
	feed_dict = {question: questions_words, text: paragraphs_sentences}
	classification = sess.run(answer_softmax, feed_dict)
	vectors = [0 for i in range(largest_num_of_words_in_answer)]
	i=0
	for word in classification:
		j=0
		emb_max = 0
		for emb in word:
			if emb > emb_max:
				emb_max = emb
				if emb_max > 0:
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








		
