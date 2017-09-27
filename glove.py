import numpy as np
import json

vocab_size = 400000

def initialise_glove_embeddings():
	glove_dimensionality = 50
	glove_path = 'glove.6B.50d.txt'
	glove_lookup = np.zeros(vocab_size, dtype='(100)string, (1,' + str(glove_dimensionality) + ')float')
	embedding_text = np.genfromtxt(glove_path, delimiter='\n', dtype='string')
	j=0
	for word_embedding_line in embedding_text:
		embeddings_for_word = word_embedding_line.split(' ')
		i=0
		embeddings_array = np.zeros(50, dtype='float64')
		for single_embedding_dimension in embeddings_for_word:
			if i > 0:
				embeddings_array[i-1] = float(single_embedding_dimension)
			i=i+1	
		glove_lookup_entry = (embeddings_for_word[0], embeddings_array)
		glove_lookup[j] = glove_lookup_entry
		j=j+1
	return glove_lookup	

#def bubble(badList):
#	length = len(badList) - 1
#	sorted = False  # We haven't started sorting yet
#	i = 0
#	while not sorted:
#		sorted = True  # Assume the list is now sorted
#		for element in range(length):
#			if badList[element][0] > badList[element + 1][0]:
#				sorted = False  # We found two elements in the wrong order
#				badList[i], badList[i+1] = badList[i+1], badList[i]
#		print str(i)
#		i = i + 1
#	return badList

glove_lookup = initialise_glove_embeddings()
#glove_lookup = bubble(glove_lookup)
	
def get_glove_embedding(word):
	embedding = 0
	for word_embedding in glove_lookup:
		if word_embedding[0] == word:
			embedding = word_embedding[1]
	return embedding	

def parse_squad():
	with open('train-v1.1.json', 'r') as squad_file:
		squad_string=squad_file.read()
		parsed_squad = json.loads(squad_string)
		return parsed_squad["data"]	

def count_squad():
	data = parse_squad()
	number_of_questions = 0
	number_of_answers = 0
	number_of_paragraphs = 0
	for text in data:
		paragraphs = text["paragraphs"]
		for paragraph in paragraphs:
			number_of_paragraphs = number_of_paragraphs + 1
			context = paragraph["context"]
			qas = paragraph["qas"]
			for qa in qas:
				number_of_questions = number_of_questions + 1
				question = qa["question"]
				answers = qa["answers"]
				for answer in answers:
					number_of_answers = number_of_answers + 1
					answer_text = answer["text"]
	return number_of_answers, number_of_questions, number_of_paragraphs

def read_squad():
	data = parse_squad()
	number_of_answers, number_of_questions, number_of_paragraphs = count_squad()
	questions_list = ['x']*number_of_questions
	answers_list = ['x']*number_of_answers
	paragraphs_list = ['x']*number_of_paragraphs
	paragraph_question_mapping = [0 for i in range(number_of_questions)]
	paragraph_num = 0
	answer_num = 0
	question_num = 0		
	for text in data:
		paragraphs = text["paragraphs"]
		for paragraph in paragraphs:
			context = paragraph["context"]
			paragraphs_list[paragraph_num] = context
			qas = paragraph["qas"]
			for qa in qas:
				question = qa["question"]
				questions_list[question_num] = question
				answers = qa["answers"]
				for answer in answers:
					answer_text = answer["text"]
					answers_list[answer_num] = answer_text
					paragraph_question_mapping[answer_num] = paragraph_num
					answer_num = answer_num + 1
				question_num = question_num + 1
			paragraph_num = paragraph_num + 1
	return questions_list, paragraphs_list, answers_list, paragraph_question_mapping

def vectorise_squad():
	questions, paragraphs, answers, paragraph_question_mapping = read_squad()
	questions = questions[:20]
	answers = answers[:20]
	paragraphs = paragraphs[:3]
	largest_num_of_sentences = 0
	largest_num_of_words = 0
	for paragraph in paragraphs:
		sentences = paragraph.split('.')
		for sentence in sentences:
			words = sentence.split(' ')
			if len(words) > largest_num_of_words:
				largest_num_of_words = len(words)
		if len(sentences) > largest_num_of_sentences:
			largest_num_of_sentences = len(sentences)
	paragraphs_sentences = [[[" " for v in range(largest_num_of_words)] for i in range(largest_num_of_sentences)] for j in range(len(paragraphs))]
	i = 0
	for paragraph in paragraphs:
		sentences = paragraph.split('.')
		j = 0
		for sentence in sentences:
			words = sentence.split(' ')
			v = 0;
			for word in words:
				print "i: " + str(i) + ",j: " + str(j) + ",v: " + str(v)
				glove_embedding = get_glove_embedding(word)	
				paragraphs_sentences[i][j][v]=glove_embedding
				v=v+1
			j=j+1
		i=i+1
	print paragraphs_sentences
	questions_words = [[" " for t in range(largest_num_of_words)] for l in range(len(questions))]	
	j = 0
	for question in questions:
		words = question.split(' ')
		v = 0;
		for word in words:
			print "j: " + str(j) + ",v: " + str(v)
			glove_embedding = get_glove_embedding(word)	
			questions_words[j][v]=glove_embedding
			v=v+1
		j=j+1	
	print questions_words
	answers_words = [[" " for p in range(largest_num_of_words)] for h in range(len(answers))]	
	j = 0
	for answer in answers:
		words = answer.split(' ')
		v = 0;
		for word in words:
			print "j: " + str(j) + ",v: " + str(v)
			glove_embedding = get_glove_embedding(word)	
			answers_words[j][v]=glove_embedding
			v=v+1
		j=j+1	
	print answers_words
	return paragraphs_sentences, questions_words, answers_words, paragraph_question_mapping

p, q, a, m = vectorise_squad()
	#gcloud ml-engine jobs submit training glove7 --module-name trainer.main --package-path Project/trainer --staging-bucket gs://fyp_neural --scale-tier BASIC --region europe-west1

