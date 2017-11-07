import tensorflow as tf
import numpy as np
import glove as glove  
import re
import string 

glove_dimensionality = 50
largest_num_of_sentences, largest_num_of_words, largest_num_of_words_any_paragraph = glove.count_words_paragraphs_in_squad()
largest_num_of_words_in_answer = glove.get_largest_num_of_words_in_answer()
print largest_num_of_sentences
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
accuracy = tf.reduce_mean(tf.cast(answer_softmax, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

paragraphs, questions, paragraph_question_mapping = glove.vectorise_squad()
a, paragraphs_str, c, paragraph_question_mapping = glove.read_squad()
with tf.Session() as sess:
	init.run()
	for i in range(1):
		answer_num = 0
		for question_x in questions:
			answer_lookup_dict = {}
			one_hot_index = 0
			sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraphs_str[paragraph_question_mapping[answer_num]])
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
			answer_one_hot[largest_num_of_words_any_paragraph-1] = 1
			answer_lookup_dict['unk'] = answer_one_hot
			answers_words = np.zeros((largest_num_of_words_in_answer, largest_num_of_words_any_paragraph))
			entry_num = 0
			#for entry in answers_words:
			#	answers_words[entry_num][largest_num_of_words_any_paragraph-1] = 1
			#	entry_num = entry_num + 1
			answerc = c[answer_num] 
			print answerc
			words = answerc.split(' ')
			v = 0;
			for word in words:
				characters = list(word)
				if len(characters) > 0:
					if characters[0] in glove.special_chars:
						try:
							glove_embedding = answer_lookup_dict[characters[0]]
						except Exception, e:
							glove_embedding = answer_lookup_dict['unk']
						answers_words[v]=glove_embedding
						v=v+1
						word = word[1:]
					if characters[len(characters)-1] in glove.special_chars:
						word = word[:-1]
					word = word.lower()
					if "'" in word and characters[0] not in "'" and characters[len(characters)-1] not in "'":
						apostrophe_word = word.split("'")
						try:
							glove_embedding = answer_lookup_dict[apostrophe_word[0]]
						except Exception, e:
							glove_embedding = answer_lookup_dict['unk']	
						answers_words[v]=glove_embedding
						v=v+1
						try:
							glove_embedding =answer_lookup_dict["'" + apostrophe_word[1]]
						except Exception, e:
							glove_embedding = answer_lookup_dict['unk']	
						answers_words[v]=glove_embedding
						v=v+1
					else:	
						try:
							glove_embedding = answer_lookup_dict[word]
						except Exception, e:
							glove_embedding = answer_lookup_dict['unk']
						answers_words[v]=glove_embedding
						v=v+1										
					if characters[len(characters)-1] in glove.special_chars:
						try:
							glove_embedding = answer_lookup_dict[characters[len(characters)-1]]
						except Exception, e:
							glove_embedding = answer_lookup_dict['unk']
						answers_words[v]=glove_embedding
						v=v+1

			an = answers_words
			qu = question_x
			pa = paragraphs[paragraph_question_mapping[answer_num]]
			print pa.shape
			sess.run(train, feed_dict={question: qu, answer: an, text: pa})
			acc_train = loss.eval(feed_dict={question: qu, answer: an, text: pa})
			print(answer_num, " Train accuracy: ", acc_train)



			feed_dict = {question: qu, text: pa}
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
			

	


			answer_num = answer_num + 1
	save_path = saver.save(sess, "/tmp/model.ckpt")


