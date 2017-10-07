import numpy as np
import json
import re
vocab_size = 400000
largest_num_of_sentences = 0
largest_num_of_words = 0
special_chars = ["'", "/", ")", "(", "/", "'", "[", "{", "]", "}", "#", "$", "%", "^", "&", "*", "-", "_", "+", "=", ".", "\""]

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

glove_lookup = initialise_glove_embeddings()
	
def get_glove_embedding(word):
	embedding = 0
	for word_embedding in glove_lookup:
		if word_embedding[0] == word:
			embedding = word_embedding[1]
			break
	return embedding

def get_one_hot_encoded_from_glove(word):
	one_hot_encoded = [0 for p in range(len(glove_lookup))]
	dimension = 0
	for word_embedding in glove_lookup:
		if word_embedding[0] == word:
			one_hot_encoded[dimension] = 1
			print str(dimension) + " " + word
			break
		dimension = dimension + 1
	return one_hot_encoded		

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
	questions_list = ['x' for i in range(number_of_questions)]
	answers_list = ['x' for i in range(number_of_answers)]
	paragraphs_list = ['x' for i in range(number_of_paragraphs)]
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

def count_words_paragraphs_in_squad():
	largest_num_of_sentences = 0
	largest_num_of_words = 0	
	questions, paragraphs, answers, paragraph_question_mapping = read_squad()
	paragraphs = paragraphs[:40]
	for paragraph in paragraphs:
		sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
		for sentence in sentences:
			words = sentence.split(' ')
			num_of_special_chars = 0;
			for word in words:
				characters = list(word)
				if characters[0] in special_chars:
					num_of_special_chars=num_of_special_chars+1
				if characters[len(characters)-1] in special_chars:
					num_of_special_chars=num_of_special_chars+1
				if "'" in word and characters[0] not in "'" and characters[len(characters)-1] not in "'":
					num_of_special_chars=num_of_special_chars+1
			if len(words) + num_of_special_chars > largest_num_of_words:
				largest_num_of_words = len(words) + num_of_special_chars
		if len(sentences) > largest_num_of_sentences:
			largest_num_of_sentences = len(sentences)
	return largest_num_of_sentences, largest_num_of_words	

count_words_paragraphs_in_squad()

def vectorise_paragraphs():
	largest_num_of_sentences, largest_num_of_words = count_words_paragraphs_in_squad()
	questions, paragraphs, answers, paragraph_question_mapping = read_squad()
	paragraphs = paragraphs[:40]
	paragraphs_sentences = np.zeros((len(paragraphs), largest_num_of_sentences, largest_num_of_words, 50))
	#paragraphs_sentences = [[[0 for v in range(largest_num_of_words)] for i in range(largest_num_of_sentences)] for j in range(len(paragraphs))]
	i = 0
	for paragraph in paragraphs:
		sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
		j = 0
		for sentence in sentences:
			words = sentence.split(' ')
			v = 0;
			for word in words:
				characters = list(word)
				if len(characters) > 0:
					if characters[0] in special_chars:
						glove_embedding = get_glove_embedding(characters[0])
						paragraphs_sentences[i][j][v]=glove_embedding
						v=v+1
						word = word[1:]
					if characters[len(characters)-1] in special_chars:
						word = word[:-1]
					word = word.lower()
					if "'" in word and characters[0] not in "'" and characters[len(characters)-1] not in "'":
						apostrophe_word = word.split("'")
						glove_embedding = get_glove_embedding(apostrophe_word[0])	
						paragraphs_sentences[i][j][v]=glove_embedding
						v=v+1
						glove_embedding = get_glove_embedding("'" + apostrophe_word[1])	
						paragraphs_sentences[i][j][v]=glove_embedding
						v=v+1
						print word + " " + apostrophe_word[0] + "'" + apostrophe_word[1]#---------------------------
					else:						
						glove_embedding = get_glove_embedding(word)	
						paragraphs_sentences[i][j][v]=glove_embedding
						v=v+1				
					if characters[len(characters)-1] in special_chars:
						glove_embedding = get_glove_embedding(characters[len(characters)-1])
						paragraphs_sentences[i][j][v]=glove_embedding
						v=v+1				
				print "i: " + str(i) + ",j: " + str(j) + ",v: " + str(v)				
			j=j+1
		i=i+1
	print paragraphs_sentences
	return paragraphs_sentences

def vectorise_questions():
	largest_num_of_sentences, largest_num_of_words = count_words_paragraphs_in_squad()
	questions, paragraphs, answers, paragraph_question_mapping = read_squad()
	questions = questions[:100]
	#questions_words = [[0 for t in range(largest_num_of_words)] for l in range(len(questions))]	
	questions_words = np.zeros((len(questions), largest_num_of_words, 50))
	j = 0
	for question in questions:
		words = question.split(' ')
		v = 0;
		for word in words:
			characters = list(word)
			if len(characters) > 0:
				if characters[0] in special_chars:
					glove_embedding = get_glove_embedding(characters[0])
					questions_words[j][v]=glove_embedding
					v=v+1
					word = word[1:]
				if characters[len(characters)-1] in special_chars:
					word = word[:-1]
				word = word.lower()
				if "'" in word and characters[0] not in "'" and characters[len(characters)-1] not in "'":
					apostrophe_word = word.split("'")
					glove_embedding = get_glove_embedding(apostrophe_word[0])	
					questions_words[j][v]=glove_embedding
					v=v+1
					glove_embedding = get_glove_embedding("'" + apostrophe_word[1])	
					questions_words[j][v]=glove_embedding
					v=v+1
					print word + " " + apostrophe_word[0] + "'" + apostrophe_word[1]#---------------------------
				else:	
					glove_embedding = get_glove_embedding(word)	
					questions_words[j][v]=glove_embedding
					v=v+1				
				if characters[len(characters)-1] in special_chars:
					glove_embedding = get_glove_embedding(characters[len(characters)-1])
					questions_words[j][v]=glove_embedding
					v=v+1				
			print "j: " + str(j) + ",v: " + str(v)
		j=j+1	
	print questions_words
	return questions_words

def vectorise_answers():#########maybe one-hot-encoded###############
	largest_num_of_sentences, largest_num_of_words = count_words_paragraphs_in_squad()
	questions, paragraphs, answers, paragraph_question_mapping = read_squad()
	answers = answers[:100]
	#answers_words = [[0 for p in range(len(glove_lookup))] for h in range(len(answers))]	
	answers_words = np.zeros((len(answers), largest_num_of_words, len(glove_lookup)))
	j = 0
	for answer in answers:
		words = answer.split(' ')
		v = 0;
		for word in words:
			characters = list(word)
			if len(characters) > 0:
				if characters[0] in special_chars:
					glove_embedding = get_one_hot_encoded_from_glove(characters[0])
					answers_words[j][v]=glove_embedding
					v=v+1
					word = word[1:]
				if characters[len(characters)-1] in special_chars:
					word = word[:-1]
				word = word.lower()
				if "'" in word and characters[0] not in "'" and characters[len(characters)-1] not in "'":
					apostrophe_word = word.split("'")
					glove_embedding = get_one_hot_encoded_from_glove(apostrophe_word[0])	
					answers_words[j][v]=glove_embedding
					v=v+1
					glove_embedding = get_one_hot_encoded_from_glove("'" + apostrophe_word[1])	
					answers_words[j][v]=glove_embedding
					v=v+1
					print word + " " + apostrophe_word[0] + "'" + apostrophe_word[1]#---------------------------
				else:	
					glove_embedding = get_one_hot_encoded_from_glove(word)	
					answers_words[j][v]=glove_embedding
					v=v+1				
				if characters[len(characters)-1] in special_chars:
					glove_embedding = get_one_hot_encoded_from_glove(characters[len(characters)-1])
					answers_words[j][v]=glove_embedding
					v=v+1				
			print "j: " + str(j) + ",v: " + str(v)
		j=j+1	
	#print answers_words
	return answers_words	

def vectorise_squad():
	a, b, c, paragraph_question_mapping = read_squad()
	return vectorise_paragraphs(), vectorise_questions(), vectorise_answers(), paragraph_question_mapping

#p, q, a, m = vectorise_squad()
	#gcloud ml-engine jobs submit training glove7 --module-name trainer.main --package-path Project/trainer --staging-bucket gs://fyp_neural --scale-tier BASIC --region europe-west1

