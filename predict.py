import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import glove as glove  
import re
import string 
glove_dimensionality = 50
largest_num_of_sentences, largest_num_of_words, largest_num_of_words_any_paragraph = glove.count_words_paragraphs_in_squad()
largest_num_of_words_in_answer = glove.get_largest_num_of_words_in_answer()
d = 400
global_step = tf.Variable(0, name="global_step")
question = tf.placeholder(tf.float32, shape=(largest_num_of_words, glove_dimensionality))
text = tf.placeholder(tf.float32, shape=(largest_num_of_sentences, largest_num_of_words, glove_dimensionality))
answer = tf.placeholder(tf.int32, shape=(largest_num_of_words_in_answer, largest_num_of_words_any_paragraph))

A = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=0.1), name="A")
B = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=0.1), name="B")
B1 = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.1), name="B1")
C = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=0.1), name="C")
W = tf.Variable(tf.random_normal([d,largest_num_of_words_any_paragraph], stddev=0.1), name="W")
X = tf.Variable(tf.random_normal([largest_num_of_words_in_answer, largest_num_of_words], stddev=0.1), name="X")
saver = tf.train.Saver()
par = "I was hired at my current workplace just over a month ago. During the interview process, I found out that someone I knew from university was also interviewing for the position. I'll call him Jim. Jim and I have never been close, but are merely aware of each other. Based solely on technical knowledge and relevant expertise, I believe that Jim would have been the better candidate for the job. In particular, he previously worked at a well known company in a position very similar to this one. I was therefore somewhat surprised when I received the offer."
par = raw_input("Enter paragraph: ")
ques = "Why were you surprised?"
ques = raw_input("Enter question: ")
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
		
#a = tf.Variable(tf.random_normal([largest_num_of_words, d], stddev=1), name="W")
u = tf.matmul(question, B1)
#u = s + a
for i in range(2):
	#print question_stacked.get_shape()
	#U = tf.matmul(question_stacked, B)
	U = tf.stack([u]*largest_num_of_sentences)
	sentence_question_match = tf.matmul(M, U, transpose_b=True)
	P = tf.nn.softmax(sentence_question_match)
	PC = tf.matmul(P, C_m)
	o = tf.reduce_sum(PC, 0)
	u = tf.add(o, u)
a = tf.matmul(X, u)
answer_hat = tf.matmul(a, W)
answer_softmax = tf.nn.softmax(logits=answer_hat)
xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=answer_hat, labels=answer)
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
a, paragraphs_str, c, paragraph_question_mapping = glove.read_squad()

with tf.Session() as sess:
	saver.restore(sess, "/tmp/model.ckpt")

	answer_lookup_dict = {}
	one_hot_index = 0
	sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', par)
	while one_hot_index < largest_num_of_words_any_paragraph - 1:
		for sentence in sentences:
			words = sentence.split(' ')
			for word in words:
				answer_one_hot = [0 for i in range(largest_num_of_words_any_paragraph)]	
				characters = list(word)
				if one_hot_index < largest_num_of_words_any_paragraph - 1:
					if len(characters) > 0:
						if characters[0] in glove.special_chars:
							if one_hot_index < largest_num_of_words_any_paragraph - 1:
								answer_one_hot[one_hot_index] = 1
								answer_lookup_dict[characters[0]] = answer_one_hot
								one_hot_index=one_hot_index+1
								answer_one_hot = [0 for i in range(largest_num_of_words_any_paragraph)]
								word = word[1:]
						if characters[len(characters)-1] in glove.special_chars:
							word = word[:-1]
						word = word.lower()
						if "'" in word and characters[0] not in "'" and characters[len(characters)-1] not in "'":
							apostrophe_word = word.split("'")
							if one_hot_index < largest_num_of_words_any_paragraph - 1:	
								answer_one_hot[one_hot_index] = 1
								answer_lookup_dict[apostrophe_word[0]] = answer_one_hot
								one_hot_index = one_hot_index + 1
								answer_one_hot = [0 for i in range(largest_num_of_words_any_paragraph)]
							if one_hot_index < largest_num_of_words_any_paragraph - 1:
								answer_one_hot[one_hot_index] = 1	
								answer_lookup_dict[apostrophe_word[1]] = answer_one_hot
								one_hot_index = one_hot_index + 1
								answer_one_hot = [0 for i in range(largest_num_of_words_any_paragraph)]
						else:	
							if one_hot_index < largest_num_of_words_any_paragraph - 1:
								answer_one_hot[one_hot_index] = 1
								answer_lookup_dict[word] = answer_one_hot
								one_hot_index = one_hot_index + 1
								answer_one_hot = [0 for i in range(largest_num_of_words_any_paragraph)]				
						if characters[len(characters)-1] in glove.special_chars:
							if one_hot_index < largest_num_of_words_any_paragraph - 1:
								answer_one_hot[one_hot_index] = 1
								answer_lookup_dict[len(characters)-1] = answer_one_hot
								one_hot_index = one_hot_index + 1
								answer_one_hot = [0 for i in range(largest_num_of_words_any_paragraph)]			
	answer_one_hot = [0 for i in range(largest_num_of_words_any_paragraph)]
	answer_one_hot[largest_num_of_words_any_paragraph-2] = 1
	answer_lookup_dict['unk'] = answer_one_hot
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
				vector = j	
			j = j + 1
		vectors[i] = vector
		i = i + 1

	print "Answer: "
	ans = ""	
	for vector in vectors:
		for word, word_embedding in answer_lookup_dict.iteritems():
			if word_embedding[vector] == 1:
				print word
				#ans = ans + " " + word
				break
	ans = ans + "."
	print ans[1:].capitalize()
		








		
