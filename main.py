import tensorflow as tf
import numpy as np
import util as util 
import re
import string

glove_dimensionality = 50
largest_num_of_sentences, largest_num_of_words, largest_num_of_words_any_paragraph = util.count_words_paragraphs_in_squad()
largest_num_of_words_in_answer = util.get_largest_num_of_words_in_answer()

d = 400
global_step = tf.Variable(0, name="global_step")
question = tf.placeholder(tf.float32, shape=(largest_num_of_words, glove_dimensionality), name="question")
text = tf.placeholder(tf.float32, shape=(largest_num_of_sentences, largest_num_of_words, glove_dimensionality), name="text")
answer = tf.placeholder(tf.int32, shape=(largest_num_of_words_in_answer, largest_num_of_words_any_paragraph+2))

A = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=0.1), name="A")
B = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=0.1), name="B")
B1 = tf.Variable(tf.random_normal([glove_dimensionality, d], stddev=0.1), name="B1")
C = tf.Variable(tf.random_normal([largest_num_of_sentences, glove_dimensionality, d], stddev=0.1), name="C")
W = tf.Variable(tf.random_normal([d,largest_num_of_words_any_paragraph+2], stddev=0.1), name="W")
X = tf.Variable(tf.random_normal([largest_num_of_words_in_answer, largest_num_of_words], stddev=0.1), name="X")

M = tf.matmul(text, A)
C_m = tf.matmul(text, C)		
u = tf.matmul(question, B1)
for i in range(3):
	U = tf.stack([u]*largest_num_of_sentences)
	sentence_question_match = tf.matmul(M, U, transpose_b=True)
	P = tf.nn.softmax(sentence_question_match)
	PC = tf.matmul(P, C_m)
	o = tf.reduce_sum(PC, 0)
	u = tf.add(o, u)
a = tf.matmul(X, u)
answer_hat = tf.matmul(a, W)
answer_softmax = tf.nn.softmax(logits=answer_hat, name="answer")
xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=answer_hat, labels=answer)
answer_c = tf.cast(answer, tf.float32)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)

#https://stackoverflow.com/questions/36498127/how-to-effectively-apply-gradient-clipping-in-tensor-flow
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 30.0)
train = optimizer.apply_gradients(zip(gradients, variables))

#train = optimizer.minimize(loss)
accuracy = tf.reduce_mean(tf.cast(answer_softmax, tf.float32))
init = tf.global_variables_initializer()

answers, paragraphs, questions, paragraph_question_mapping = util.vectorise_squad()
_, paragraphs_str, _, _ = util.read_squad()
with tf.Session() as sess:
	init.run()
	for i in range(10):
		answer_num = 0
		for question_x in questions:	
			an = answers[answer_num]
			qu = question_x
			pa = paragraphs[paragraph_question_mapping[answer_num]]
			sess.run(train, feed_dict={question: qu, answer: an, text: pa})
			acc_train = loss.eval(feed_dict={question: qu, answer: an, text: pa})
			#if(answer_num%100==0):
			print(i ,answer_num, " Train accuracy: ", acc_train)
			#if(answer_num%1000==0):
				#builder = tf.saved_model.builder.SavedModelBuilder("model" + str(i))
				#builder.add_meta_graph_and_variables(sess, ["tag"], signature_def_map= {
				#	"model": tf.saved_model.signature_def_utils.predict_signature_def(
				#	    inputs= {"question": question, "text": text},
				#	    outputs= {"answer": answer_softmax})
				#	})
				#builder.save()
			feed_dict = {question: qu, text: pa}
			classification = sess.run(answer_softmax, feed_dict)
			answer_lookup_dict = util.get_answer_dictionary(answer_num, paragraphs_str, paragraph_question_mapping, largest_num_of_words_any_paragraph+1)
			print util.get_words(classification, answer_lookup_dict, largest_num_of_words_in_answer)
			answer_num = answer_num + 1

	#validate
	answers, paragraphs, questions, new_paragraph_question_mapping = util.vectorise_validation_squad()
	_, paragraphs_str, _, paragraph_question_mapping = util.read_squad()
	answer_num = 0
	absolute_answer_num = 77000
	overall_accuracy = 0
	for question_x in questions:	
		an = answers[answer_num]
		qu = question_x
		pa = paragraphs[new_paragraph_question_mapping[answer_num]]
		feed_dict = {question: qu, text: pa}
		classification = sess.run(answer_softmax, feed_dict)
		answer_lookup_dict = util.get_answer_dictionary(absolute_answer_num, paragraphs_str, paragraph_question_mapping, largest_num_of_words_any_paragraph+1)
		words_answer_hat = util.get_words(classification, answer_lookup_dict, largest_num_of_words_in_answer)
		words_answer = util.get_words(an, answer_lookup_dict, largest_num_of_words_in_answer)
		print words_answer_hat
		print words_answer
		split_hat = words_answer_hat.split(' ')
		split = words_answer.split(' ')
		correct_words_count = 0
		for word in split:
			for word_hat in split_hat:
				if word_hat==word:
					correct_words_count = correct_words_count + 1
		accuracy = float(correct_words_count)/float(len(split))
		print ("Example accuracy: ", accuracy)
		overall_accuracy = (accuracy+float(overall_accuracy))/float(answer_num + 1)

		print ("Overall accuracy: ", overall_accuracy)
		answer_num = answer_num + 1	
		absolute_answer_num = absolute_answer_num + 1
