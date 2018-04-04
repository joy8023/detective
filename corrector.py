import tensorflow as tf
import numpy as np
import keras as k
import random
import pickle
import time

# object in pkl form from test_attack.py
# original image and label
# adversary image and label
class newdata:
    def __init__(self, od, ol, ad, al):
        self.origin_data = np.array(od)
        self.origin_label= np.array(ol)
        self.adv_data = np.array(ad)
        self.adv_label= np.array(al)

# object for training or testing contains data and corresponding labels
# num means number of good
class DataSet:
	def __init__(self,data,label,num):
		self.data = data
		self.label = label
		self.num = num

	def dstl(self):
		self.data = self.data/100

class Corrector:
	def __init__(self, model, datapath, test, target = True, r = 0.02, n = 50):
		self.model = model
		self.path = datapath
		self.test = test
		self.r = r
		self.n = n
		if target:
			self.target = 9
		else:
			self.target = 1

	def load_data_region(self, filepath, false):
		f = open(filepath,'rb')
		newdata = pickle.load(f)
		f.close

		true_label = np.argmax(np.squeeze(newdata.origin_label),axis = 1)

		good_data = []
		good_label = []
		adv_data = []
		adv_label = []
		diff = []
		good_num = newdata.origin_label.shape[0]

		for i in false.tolist():
			if i < good_num:
				good_data.append(newdata.origin_data[i])
				good_label.append(true_label[i])
			else:
				origin_idx = int((i-good_num)/self.target)
	#			print(origin_idx)
				adv_data.append(newdata.adv_data[i-good_num])
				adv_label.append(true_label[origin_idx])
				diff.append(np.sum((newdata.adv_data[i-good_num]-newdata.origin_data[origin_idx])**2)**.5)
	#			print(diff)

		good = DataSet(good_data,good_label,len(good_label))
		adv = DataSet(adv_data,adv_label,len(adv_label))
		print('good data:',good.num)
		print('adv data:',adv.num)
		print('total mean noise:',np.sum(diff)/len(diff))

		return good, adv, diff


	def sample(self, input):
		#width, height, channel 
		w, h, c = input.shape
		output = np.tile(input,(self.n,1,1,1))

		for i in range(w):
			for j in range(h):
				for k in range(c):
					lower = max(-0.5,input[i][j][k]-self.r)
					upper = min(0.5,input[i][j][k]+self.r)
					for l in range(self.n):
						output[l][i][j][k] = random.uniform(lower, upper)

		return output

	def evaluate(self, model, dataset, diff, noise = False):
		data = dataset.data
		label = dataset.label
		num = dataset.num

		correct = []
		idx = []
		error = 0
		for i in range(num):
			samples = self.sample(data[i])
#			pred_label = predict(model, samples)
#			correct.append(pred_label)
			prob = model.model.predict(samples)
			cla = np.argmax(prob, axis = 1)
			count = np.bincount(cla)
			pred_label = np.argmax(count)
			if pred_label != label[i]:
				error += 1
				idx.append(i)

		if noise and (len(idx) != 0):
#			print(idx)
			noises = 0
			for item in idx:
				noises += diff[item]
			mean = noises/len(idx)
			print('success mean noise:', mean)

		accuracy = 1 - error/num
		if noise:
			print('adv error:', error)
		else:
			print('good error:', error)
#		print('accuracy:', accuracy)
		return error

	def correct(self):
		good, adv, diff = self.load_data_region(self.path, self.test)

		err_good = self.evaluate(self.model, good, diff)
		err_adv = self.evaluate(self.model, adv, diff,True)


		return err_good, err_adv


'''
def load_data_region(filepath, false):
	f = open(filepath,'rb')
	newdata = pickle.load(f)
	f.close

	true_label = np.argmax(np.squeeze(newdata.origin_label),axis = 1)

	good_data = []
	good_label = []
	adv_data = []
	adv_label = []
	good_num = newdata.origin_label.shape[0]
	for i in false.tolist():
		if i < good_num:
			good_data.append(newdata.origin_data[i])
			good_label.append(true_label[i])
		else:
			adv_data.append(newdata.adv_data[i-good_num])
			adv_label.append(true_label[int((i-good_num)/9)])

	good = DataSet(good_data,good_label,len(good_label))
	adv = DataSet(adv_data,adv_label,len(adv_label))
	print(good.num)
	print(adv.num)

	return good, adv


def sample(input,r = 0.1, n = 50):
	#widthm height, channel 
	w, h, c = input.shape
	output = np.tile(input,(n,1,1,1))

	for i in range(w):
		for j in range(h):
			for k in range(c):
				lower = max(-0.5,input[i][j][k]-r)
				upper = min(0.5,input[i][j][k]+r)
				for l in range(n):
					output[l][i][j][k] = random.uniform(lower, upper)

	return output

def evaluate(model, dataset):
	data = dataset.data
	label = dataset.label
	num = dataset.num

	correct = []
	error = 0
	for i in range(num):
		samples = sample(data[i])
#		pred_label = predict(model, samples)
#		correct.append(pred_label)
		prob = model.model.predict(samples)
		cla = np.argmax(prob, axis = 1)
		count = np.bincount(cla)
		pred_label = np.argmax(count)
		if pred_label != label[i]:
			error += 1

	accuracy = 1 - error/num
	print('error:', error)
	print('accuracy:', accuracy)

#test false negatives on all data
def testgood(model,test):
	num = test.num

	good_data = test.data[:num,]
	good_label = test.label[:num,]

	adv_data = test.data[num:,]
	adv_label = test.label[num:,]

	score_good = model.evaluate(good_data,good_label,verbose = 0)
	score_adv = model.evaluate(adv_data,adv_label,verbose = 0)

	mnist = (score_good[1]+(1-score_adv[1])*11)/12
	cifar = (score_good[1]+(1-score_adv[1])*9)/10
	print('mnist:', mnist)
	print('cifar:', cifar)


def correct(false):
	good,adv = load_data_region('data/mnist/dstl/100start0.pkl', false)
	region_model = MNISTModel("models/mnist-distilled-100")

	evaluate(region_model, good)
	evaluate(region_model, adv)
'''