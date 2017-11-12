import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import glove as glove  
import re
import string 
largest_num_of_sentences, largest_num_of_words, largest_num_of_words_any_paragraph = glove.count_words_paragraphs_in_squad()
largest_num_of_words_in_answer = glove.get_largest_num_of_words_in_answer()

def get_words(classification):
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

	ans = ""	
	for vector in vectors:
		for word, word_embedding in answer_lookup_dict.iteritems():
			if word_embedding[vector] == 1:
				try:
					ans = ans + " " + word
				except Exception:
					ans = ans + " " + str(word)
				break
	ans = ans + "."
	return ans[1:].capitalize()

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

a, paragraphs_str, c, paragraph_question_mapping = glove.read_squad()

with tf.Session(graph=tf.Graph()) as sess:
	tf.saved_model.loader.load(sess, ["tag"], "model")
	graph = tf.get_default_graph()
	question = graph.get_tensor_by_name("question:0")
	text = graph.get_tensor_by_name("text:0")
	answer_softmax = graph.get_tensor_by_name("answer:0")

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
	print get_words(classification)
		








		
