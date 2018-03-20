## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time
import random
import pickle

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
#from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi

class newdata:
    def __init__(self, od, ol, ad, al):
        self.origin_data = np.array(od)
        self.origin_label= np.array(ol)
        self.adv_data = np.array(ad)
        self.adv_label= np.array(al)

        print(self.origin_data.shape)
#        print(self.origin_label.shape)
        print(self.adv_data.shape)
#        print(self.adv_label.shape)

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

def generate_data(model, data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if np.argmax(model.model.predict(data.train_data[start+i:start+i+1])) == np.argmax(data.train_labels[start+i]):
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.train_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.train_data[start+i])
                targets.append(np.eye(data.train_labels.shape[1])[j])
#        else:
#            inputs.append(data.test_data[start+i])
#            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)
    print(inputs.shape[0]/9)

    return inputs, targets

def evaluate(model, data, label):
    predict = model.predict(data)
    pl = np.argmax(predict,axis = 1) 
    l = np.argmax(label,axis = 1)
    error = 0
    num = pl.shape[0]
    for i in range(num):
        if pl[i] != l[i]:
            error +=1

    accuracy = 1 - float(error)/num
    print('accuracy:',accuracy)

    return accuracy

origin_data = []
adv_data = []
origin_label = []
adv_label = []
ut_data = []
ut_label = []

samples = 1001
start = 1000
confidence = 0

filename = 'dstl_mnist1000start1000.pkl'
utfile = 'ut_'+filename

if __name__ == "__main__":
    with tf.Session() as sess:

#        data, model =  MNIST(), MNISTModel("models/mnist", sess)
        data, model =  MNIST(), MNISTModel("models/mnist-distilled-100", sess)
#        data, model =  CIFAR(), CIFARModel("models/cifar", sess)
#        data, model =  CIFAR(), CIFARModel("models/cifar-distilled-100", sess)
        
#        evaluate(model.model, data.train_data, data.train_labels)

        attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=confidence)
#        attack = CarliniL0(sess, model)
#        attack = CarliniLi(sess, model)

        inputs, targets = generate_data(model, data, samples=samples, targeted=True,
                                        start=start, inception=False)

        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()

        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

        for i in range(int(len(adv)/9)):
            origin_data.append(inputs[i*9])
            origin_label.append(model.model.predict(inputs[i*9:i*9+1]))
            min = np.sum((adv[i*9]-inputs[i*9])**2)
            idx=0
            for j in range(9):
                adv_data.append(adv[i*9+j])
                adv_label.append(model.model.predict(adv[i*9+j:i*9+j+1]))
                dist = np.sum((adv[i*9+j]-inputs[i*9+j])**2)
                if dist<min:
#                    print(dist)
                    min = dist
                    idx = j

            ut_data.append(adv[i*9+idx])
            ut_label.append(model.model.predict(adv[i*9+idx:i*9+idx+1]))
#            show(adv[i*9+idx])

'''
        for i in range(len(adv)):
#            print("Valid:")
#            show(inputs[i])
#            print("Adversarial:")
#            show(adv[i])

#            print("Classification:", model.model.predict(adv[i:i+1]))
#            print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)
'''
new_data=newdata(origin_data, origin_label, adv_data, adv_label )
new_ut = newdata(origin_data, origin_label, ut_data, ut_label)
#print(ut_label)

f = open(filename,'wb')
pickle.dump(new_data,f)
f.close

f = open(utfile,'wb')
pickle.dump(new_ut,f)
f.close

