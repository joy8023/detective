import tensorflow as tf
import keras as k
import numpy as np
import copy
import pickle
import time
import random

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

from setup_cifar import CIFARModel
from setup_mnist import MNISTModel
from corrector import Corrector, newdata, DataSet

def load_data_crop(filepath, n):
	f = open(filepath,'rb')
	newdata = pickle.load(f)
	f.close

	data = np.concatenate((np.squeeze(newdata.origin_label[0:n]),
						   np.squeeze(newdata.adv_label[0:n*9])),
						  axis=0)

	good_label = [[1,0]]
	adv_label = [[0,1]]

	good_num = n
	adv_num = n*9

	label = good_label*good_num+adv_label*adv_num
	label = np.array(label)

	return DataSet(data, label, good_num)

def load_data(filepath):
	f = open(filepath,'rb')
	newdata = pickle.load(f)
	f.close

	data = np.concatenate((np.squeeze(newdata.origin_label),
						   np.squeeze(newdata.adv_label)),
						  axis=0)

	good_label = [[1,0]]
	adv_label = [[0,1]]

	good_num = newdata.origin_label.shape[0]
	adv_num = newdata.adv_label.shape[0]

	label = good_label*good_num+adv_label*adv_num
	label = np.array(label)

	return DataSet(data, label, good_num)

def binary_model(train, test):

	t1 = time.time()

	model = Sequential()

	model.add(Dense(16, input_dim=10, activation='relu'))
	model.add(Dense(32,activation='relu'))
	model.add(Dense(32,activation='relu'))
#	model.add(Dense(16,activation='relu'))
	model.add(Dense(2, activation='sigmoid'))

	optimizer = k.optimizers.SGD(lr= 0.01,
								 momentum = 0.5)

	model.compile(optimizer=optimizer,
				  loss ='binary_crossentropy',
				  metrics=['accuracy'])

	model.fit(train.data, train.label,
			  batch_size = 8,
		      epochs = 20,
		      verbose = 1,
		      shuffle = True)

	t2 = time.time()

	score = model.evaluate(test.data,test.label,verbose = 0)
#	print('overall loss:', score[0])
	print('**********detector************')
	print('overall accuracy:', score[1])
	
	num = test.num

	good_data = test.data[:num,]
	good_label = test.label[:num,]

	adv_data = test.data[num:,]
	adv_label = test.label[num:,]

	score_good = model.evaluate(good_data,good_label,verbose = 0)
#	print('good data loss:', score_good[0])
	print('good data accuracy:', score_good[1])

	score_adv = model.evaluate(adv_data,adv_label,verbose = 0)
#	print('adv data loss:', score_adv[0])
	print('adv data accuracy:', score_adv[1])

	print("time to train model on %i good data:"%train.num, t2- t1)
	
	return model, score_adv[1]

def detect(m,test):
	model = m
	tt1 = time.time()
	predict = model.predict(test.data)
	tt2 = time.time()
	right = predict.argmax(axis=1)
	false = np.where(right == 1)[0]
	print('time to detect:', tt2-tt1)

	return false

def DCN(trainpath,testpath,modelpath,dstl = False, target = True):

	train = load_data(trainpath)
	test = load_data(testpath)
	#good = load_data('data/mnist5kgood55k.pkl')

	if dstl:
#		train.dstl()
		test.dstl()

	if target:
		para = 9
	else:
		para = 1

	model, adv_accu = binary_model(train, test)
	false = detect(model,test)
	print('**********corrector************')
	region_model = CIFARModel(modelpath)
	t5 = time.time()
	c = Corrector(region_model, testpath, false, target = target,r=0.02, n = 50)
	error = c.correct()
	t6 = time.time()

	accuracy_good = (test.num - error[0])/test.num
	attack_success = (1-adv_accu)+error[1]/test.num/para
	print('accuracy_good:',accuracy_good)
	print('attack_success:',attack_success)
	print('time:', t6 -t5)
	#model = MNISTModel("models/mnist")
	#testgood(model, good)


trainpath = 'data/cifar/500start500.pkl'
testpath = 'data/cifar/ut_l0_100start0.pkl'
modelpath = "models/cifar"
dstl = False
target = False

DCN(trainpath, testpath, modelpath, dstl = dstl, target = target)