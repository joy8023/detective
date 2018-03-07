import tensorflow as tf
import keras as k
import numpy as np
import copy
import pickle
import time

from sklearn import preprocessing

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

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

# containing data of the same predicted label
class OneClass:
	def __init__(self, classes):
		self.classes = classes
		self.data = []
		self.label = []
		self.num = 0  #number of good data

	def add_data(self, data):
		self.data.append(data)

	def add_label(self, label):
		self.label.append(label)

	def trans_array(self):
		self.data = np.array(self.data)
		self.label = np_utils.to_categorical(self.label)

	# add data of another class to this class 
	def add_class(self, oneclass):
		self.classes = -1
		self.data = np.concatenate((self.data,oneclass.data),axis=0)
		self.label = np.concatenate((self.label,oneclass.label),axis=0)
		self.num = self.num + oneclass.num

	def add_good(self):
		self.num = self.num + 1

#split data into ten classes according to their predict label
#binary to be false means each data will have their own right label
#binary to be true means each data will be marked good or adversary

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

def split_data(dataset,binary = False):
	num = dataset.num

	# this list contains ten oneclass object for each classification
	classes = []
	for i in range(10):
		classes.append(OneClass(i))

	#predict is the label given the model
	predict = dataset.data.argmax(axis = 1)

	#split for good data with the same predict label
	for i in range(num):
		predict_label = predict[i]
		classes[predict_label].add_data(dataset.data[i])

		classes[predict_label].add_good()
		if binary == False:
			classes[predict_label].add_label(predict_label)
		else:
			classes[predict_label].add_label(0)
	
	adv_data = dataset.data[num:,]

	# split for adversary data with the right label
	for i in range(adv_data.shape[0]):
		predict_label = predict[i+num]
		classes[predict_label].add_data(adv_data[i])
		if binary == False:
			#corresponding good data's label
			right_label = predict[int(i/9)]
			classes[predict_label].add_label(right_label)
		else:
			classes[predict_label].add_label(1)

	for i in range(10):
		classes[i].trans_array()

	return classes

def binary_model(train, test):

	t1 = time.time()

	model = Sequential()

	model.add(Dense(16, input_dim=10, activation='relu'))
	model.add(Dense(32,activation='relu'))
#	model.add(Dense(32,activation='relu'))
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
		      verbose = 0,
		      shuffle = True)

	t2 = time.time()

	score = model.evaluate(test.data,test.label,verbose = 0)
#	print('overall loss:', score[0])
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

	print("time to train model on %i data:"%(train.num*10), t2- t1)
	
	return model

def ten_model(train, test):

	model = Sequential()

	model.add(Dense(32, input_dim=10, activation='relu'))
	model.add(Dense(64,activation='relu'))
	model.add(Dense(64,activation='relu'))
	model.add(Dense(32,activation='relu'))
	model.add(Dense(10, activation='softmax'))

	optimizer = k.optimizers.SGD(lr= 0.01,
								 momentum = 0.2)

	model.compile(optimizer='Nadam',
				  loss ='categorical_crossentropy',
				  metrics=['accuracy'])

	model.fit(train.data, train.label,
			  batch_size = 4,
		      epochs = 20,
		      verbose = 0,
		      shuffle = True)

	score = model.evaluate(test.data,test.label,verbose = 0)
#	print('test loss:', score[0])
#	print('test accuracy:', score[1])

	return score,model

def print_false(m,test):
	model = m

	predict = model.predict(test.data)
	right_label = test.label.argmax(axis=1)
#	print(right_label)
	pre_label = predict.argmax(axis=1)
#	print(pre_label)

	for i in range(len(right_label)):
		if right_label[i] != pre_label[i]:
			print(right_label[i], pre_label[i], test.data[i])

def ten_train(train, test):

	train_all = split_data(train)
	test_all = split_data(test)

	score = []
	for i in range(10):
		s, model = ten_model(train_all[i],test_all[i])
		score.append(s)
		print('class:%d accuracy:%f' % (i,score[i][1]))
	#	print_false(model, test_all[i])

'''
	#train data together
	class10 = copy.deepcopy(train_all[0])
	class10_test = copy.deepcopy(test_all[0])
	for i in range(1,10):
		class10.add_class(train_all[i])
		class10_test.add_class(test_all[i])
	score,model = ten_model(class10,class10_test)
	print('accuracy:%f' % (score[1]))
'''
#train on each class for binary classification
def binary_train(train, test):

	train_all = split_data(train, True)
	test_all = split_data(test, True)

	score = []
	for i in range(10):
		print('class',i)
		model = binary_model(train_all[i],test_all[i])
#		print_false(model, test_all[i])

#train = load_data('data/cifar500start0.pkl')
#test = load_data('data/cifar500start500.pkl')
train = load_data('data/mnist100start0.pkl')
test = load_data('data/mnist1000start1000.pkl')

#m = binary_model(train, test)
#print_false(m, test)

model = binary_model(train, test)

t1 = time.time()
model = model.predict(test.data)
t2 = time.time()
print("for %i data, time: "%(test.num*10),t2-t1)
#print_false(model, test)

#binary_train(train,test)

#b=tf.nn.softmax(a)
#with tf.Session() as sess:

#        print(sess.run(b))
