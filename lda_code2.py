import numpy as np
from scipy.special import gamma
from scipy.special import digamma
from scipy.special import gammaln
from operator import itemgetter
from collections import OrderedDict
import time
import math
import random

class Document():
	def __init__(self, length):
		self.length = length
		self.words = np.zeros(length, dtype = np.int32)
		self.counts = np.zeros(length, dtype = np.int32)
		self.word_total_number = 0
class Corpus():
	def __init__(self, docs):
		self.docs = docs
		self.number_docs = len(docs)
class LDA():
	LAG = 1
	INITIAL_ALPHA = 0.1
	NTOPICS = 10
	EM_CONVERGED = 1e-13
	EM_MAX_ITER = 30
	VAR_CONVERGED = 1e-9
	VAR_MAX_ITER = 30
	NUM_INIT = 1
	def __init__(self, alpha, number_topics, number_terms):
		self.number_topics = number_topics
		self.number_terms= number_terms
		if alpha != None:
			self.alpha = alpha
		else:
			self.alpha = self.INITIAL_ALPHA
		self.log_beta = np.zeros(number_topics*number_terms, dtype = "float64").reshape(number_topics, number_terms)
	def maximumLikelihood(self):
		def log_sum(log_a, log_b):
			if log_a < log_b:
				return log_b + math.log(1+ math.exp(log_a - log_b))    
			else:
				return log_a + math.log(1+ math.exp(log_b - log_a))
		def inference(doc, phi):
			def compute_likelihood(doc, phi, var_gamma, var_dig_gamma, log_beta_NxK):
				var_gamma_sum = np.sum(var_gamma)
				likelihood = gammaln(self.alpha * self.number_topics)\
							- self.number_topics * gammaln(self.alpha)\
							- gammaln(var_gamma_sum)
				likelihood += np.sum((self.alpha - var_gamma + np.sum((doc.counts * phi.T).T, axis = 0)) \
									*(var_dig_gamma - digamma(var_gamma_sum)) \
									+ gammaln(var_gamma) \
									+ np.sum((doc.counts * phi.T).T * (log_beta_NxK - np.log(phi)), axis = 0))
				return likelihood 
			# Init var_gamma
			var_gamma = np.empty(self.number_topics, dtype="float64")
			var_gamma.fill(self.INITIAL_ALPHA + doc.word_total_number)
			var_dig_gamma = digamma(var_gamma)
			# Caculate log_beta for doc
			log_beta_NxK = np.zeros(doc.length*self.number_topics, dtype = "float64")\
							.reshape(doc.length, self.number_topics)
			for n in range(doc.length):
				for k in range(self.number_topics):
					log_beta_NxK[n][k] = self.log_beta[k][doc.words[n]]
			# Update phi, var_gamma for doc
			var_iter = 0
			old_phi = np.zeros(self.number_topics, dtype ="float64")
			converged = 1.0
			old_likelihood = 0.0
			while converged  > self.VAR_CONVERGED and (var_iter < self.VAR_MAX_ITER or self.VAR_MAX_ITER == -1) :
				# print(var_iter)
				var_iter += 1
				old_phi = phi.copy()
				# Update phi
				phi[:][:] = np.exp(var_dig_gamma + log_beta_NxK)
				phisum = np.sum(phi, axis =1)
				new_phi = (phi.T / phisum).T
				phi[:][:] = new_phi
				# phi = var_dig_gamma + log_beta_NxK
				# phisum = np.zeros(doc.length)
				# for n in range(doc.length):
				# 	for k in range(self.number_topics):
				# 		phisum[n] = log_sum(phi[n][k], phisum[n])
				# phi = np.exp((phi.T - phisum).T)	

				#Update var_gamma			
				var_gamma += np.sum(doc.counts*(phi - old_phi).T, axis = 1)
				var_dig_gamma = digamma(var_gamma)
				# Caculate likelihood
				likelihood = compute_likelihood(doc, phi, var_gamma, var_dig_gamma, log_beta_NxK)
				if old_likelihood != 0:
					converged = (old_likelihood - likelihood)/old_likelihood
					# print("cove = {}".format(converged))
				old_likelihood = likelihood
			return likelihood
		def doc_e_step(doc, sum_prob_z_when_w_in_copurs):
			phi = np.empty((doc.length, self.number_topics),dtype="float64")
			phi.fill(1/self.number_topics)
			# Caculate likelihood of doc (inference)
			likelihood = inference(doc, phi)
			# Valuate sum_prob_z_when_w_in_copurs
			for n in range(doc.length):
				for k in range(self.number_topics):
					sum_prob_z_when_w_in_copurs[k][doc.words[n]] += doc.counts[n]*phi[n][k]
			return likelihood
		
		
				   
		def caculate_log_beta(sum_prob_z_when_w_in_copurs):
			return (np.log(sum_prob_z_when_w_in_copurs).T - np.log(np.sum(sum_prob_z_when_w_in_copurs, axis = 1))).T
		def lda_mle(sum_prob_z_when_w_in_copurs):
			self.log_beta = caculate_log_beta(sum_prob_z_when_w_in_copurs)
		def init_log_beta(corpus, num_init,sum_prob_z_when_w_in_copurs):
			seen = [[0]*num_init]*self.number_topics
			d = 0
			for k in range(self.number_topics):
				for	i in range(num_init):
					while True:
						d = math.floor(random.random() * (corpus.number_docs - 1))
						already_selected = 0
						for j in range(k):
							if seen[j][i] == d:
								already_selected = 1
						if already_selected == 0: break
					seen[k][i] = d
					doc = corpus.docs[d]
					for n in range(doc.length):
						sum_prob_z_when_w_in_copurs[k][doc.words[n]] += doc.counts[n]
				for nj in range(self.number_terms):
						sum_prob_z_when_w_in_copurs[k][nj] += 1	
			self.log_beta = caculate_log_beta(sum_prob_z_when_w_in_copurs)
			
		def readData(file_name):
			docs = []
			file = open(file_name)
			while True:
				line = file.readline()
				if line == "": break  
				list_doc_presentation = line.split(" ")
				doc = Document(int(list_doc_presentation[0]))
				for i in range(len(list_doc_presentation)-1):
					word_presentation = list_doc_presentation[i+1].split(":")
					doc.words[i] = int(word_presentation[0])
					doc.counts[i] = int(word_presentation[1])
				doc.word_total_number = np.sum(doc.counts)
				docs += [doc]
			corpus = Corpus(docs)
			return corpus
		##############
		# init for copurs and beta
		corpus = readData("ap.dat")
		sum_prob_z_when_w_in_copurs = np.zeros(self.number_topics*self.number_terms, dtype = "float64")\
										.reshape(self.number_topics,self.number_terms)
		init_log_beta(corpus, self.NUM_INIT,sum_prob_z_when_w_in_copurs)
		# EM Algorithm
		old_likelihood = 0.0
		converged = 1.0
		i = 0
		while ((converged < 0 or converged > self.EM_CONVERGED) and i <= self.EM_MAX_ITER):
			start = time.time()
			sum_prob_z_when_w_in_copurs.fill(0.0)
			#e step
			likelihood = 0.0
			for d in range(corpus.number_docs):
				likelihood += doc_e_step(corpus.docs[d],sum_prob_z_when_w_in_copurs)

			#m step
			lda_mle(sum_prob_z_when_w_in_copurs)
			if i >0:
				converged = (old_likelihood - likelihood)/old_likelihood			
			old_likelihood = likelihood
			if i% self.LAG == 0:
				print("Lap lan {}, converged = {}, likelihood = {}, Time = {}".format(i,converged, likelihood,time.time() - start))
			i += 1
		
def readVocab(file_name):
	file = open(file_name)
	vocabs = file.read().split("\n")
	return vocabs

def saveWordOfEachTopic(file_name):
	result_file = open(file_name, "w")
	result_file.truncate()
	for k in range(lda.number_topics):
		dict = {i: lda.log_beta[k][i] for i in range(lda.number_terms)}
		sorted_dict = OrderedDict(sorted(dict.items(), key=itemgetter(1)))
		words = []
		i = 0
		for key in sorted_dict.keys():
			words += [vocabs[key]]
			i+=1
			if i >10: break
		string = ""
		for word in words:
			string += word + ", "
		print("Chu de {} :{}".format(k,string))
		result_file.write(string)
		result_file.write('\n')
	result_file.close()
if __name__ == '__main__':
	vocabs = readVocab("vocab.txt")
	number_topics = 50
	number_terms = 10473
	# lda
	lda = LDA(None,number_topics,number_terms)
	lda.maximumLikelihood()
	# Saving result
	saveWordOfEachTopic("result.txt")
	


					

		  