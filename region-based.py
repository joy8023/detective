import tensorflow as tf
import numpy as np
import keras as k
import random
import pickle
import copy
import time

from setup_cifar import CIFARModel
from setup_mnist import MNISTModel

# object in pkl form from test_attack.py
# original image and label
# adversary image and label
class newdata:
    def __init__(self, od, ol, ad, al):
        self.origin_data = np.array(od)
        self.origin_label= np.array(ol)
        self.adv_data = np.array(ad)
        self.adv_label= np.array(al)

class DataSet:
	def __init__(self,data,label,num):
		self.data = data
		self.label = label
		self.num = num

	def add_prob(self, prob):
		self.prob = prob

def load_data(filepath):
	f = open(filepath,'rb')
	newdata = pickle.load(f)
	f.close

	#good data
	od = newdata.origin_data

	sqz = np.squeeze(newdata.origin_label)
	if len(sqz.shape) > 1:
		ol = np.argmax(sqz,axis = 1)
	else:
		ol = np.argmax(sqz)
		ol = np.array([ol])

	good = DataSet(od, ol, ol.shape[0])

	#adversary data
	#get the true label and repeat for 9 times of adversary data
	ad = newdata.adv_data
	label = ol.repeat(9)
#	label = ol
	adversary = DataSet(ad, label, label.shape[0])	
	adversary.add_prob(newdata.adv_label)
	al = np.squeeze(newdata.adv_label)
	
	return  good, adversary, al

def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

#input is an image channel last like 28*28*1
#r means 
#n is the numbers of sample for every input
def sample(input,r = 0.3, n = 1000):
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

#input: samples for one data
#output: the most frequent label
def predict(model, data):
		
	prob = model.model.predict(data)
	#the classification for each sample
	cla = np.argmax(prob, axis = 1)
	#count the number of classification
	count = np.bincount(cla)
	#get the most frequent one
	label = np.argmax(count)

	return label

def evaluate(model, dataset, advl):
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

def test_dstl(model, test):
	 model = model
	 predict = model.model.predict(test.data)
	 print(np.argmax(predict, axis = 1))
	 print(test.label)


#model = CIFARModel("models/cifar")
#fp = 'data/cifar/ut_l0_100start0.pkl'

model = MNISTModel("models/mnist")
fp = 'data/mnist/li_100start0.pkl'

good,adv,advl= load_data(fp)

#time1 = time.time()
#evaluate(model, good,advl)
#time2 = time.time()
#print('for %i good data, time is :' %good.num, time2-time1)

time3 = time.time()
evaluate(model, adv,advl)
time4 = time.time()
print('for %i adversary data, time is : ' %adv.num, time4-time3)

