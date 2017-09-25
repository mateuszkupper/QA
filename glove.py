import numpy as np
import json

vocab_size = 400000

def initialise_glove_embeddings():
	glove_dimensionality = 50
	glove_path = '/media/mateusz/Windows/Users/mateu/Downloads/glove.6B/glove.6B.50d.txt'
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

glove_lookup = initialise_glove_embeddings()

def get_glove_embedding(word):
	embedding = 0
	for word_embedding in glove_lookup:
		if word_embedding[0] == word:
			embedding = word_embedding[1]
	return embedding

def parse_squad():
	with open('/home/mateusz/Downloads/train-v1.1.json', 'r') as squad_file:
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
	paragraph_question_mapping = [0]*number_of_questions
	paragraph_num = 0
	answer_num = 0
	question_num = 0		
	for text in data:
		paragraphs = text["paragraphs"]
		for paragraph in paragraphs:
			context = paragraph["context"]
			print context
			print paragraph_num
			print number_of_paragraphs
			paragraphs_list[paragraph_num] = context
			qas = paragraph["qas"]
			for qa in qas:
				question = qa["question"]
				print "Q: " + question
				print question_num
				print number_of_questions
				questions_list[question_num] = question
				answers = qa["answers"]
				for answer in answers:
					answer_text = answer["text"]
					print answer_text
					print answer_num
					print number_of_answers
					answers_list[answer_num] = answer_text
					paragraph_question_mapping[answer_num] = paragraph_num
					answer_num = answer_num + 1
				question_num = question_num + 1
			paragraph_num = paragraph_num + 1
	print "Answers: " + str(number_of_answers)
	print "Questions: " + str(number_of_questions)
	print "Paragraphs: " + str(number_of_paragraphs)
	return questions_list, paragraphs_list, answers_list

def vectorise_squad():
	questions, paragraphs, answers = read_squad()
	x = 0 #largest number of sentences in a paragraph
	for paragraph in paragraphs:
		sentences = paragraph.split('.')
		if len(sentences) > x:
			x = len(sentences)
	paragraphs_sentences = [['x'] for i in range(x)]*(len(paragraphs))
	i = 0
	for paragraph in paragraphs:
		sentences = paragraph.split('.')
		j = 0
		for sentence in sentences:
			paragraphs_sentences[i][j]=sentence
			j=j+1
		i = i+1
	print paragraphs_sentences
	

