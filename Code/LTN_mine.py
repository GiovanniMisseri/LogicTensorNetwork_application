# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:17:48 2018

@author: luigimissri
"""
import pandas as pd

dat=pd.read_csv("C:/Users/luigimissri/Desktop/FILE DA SALVARE/Data Science/Knoledge and Data Mining/Progetto/dati_anal.csv")
del dat["Unnamed: 0"]
#identificativo=dat["id"]
#del dat["id"]


import numpy as np
from itertools import combinations

dati=np.array(dat)

alpha = 0.40
beta = 0.90
observation,variables = dati.shape
Omega = range(variables)
transactions = range(observation)

Fi=np.sum(dati,axis=0)


items=list(dat.columns.values)


def get_frequent_itemsets(dati,alpha):
    print(dati)
    L = {0:{tuple():observation}}
    Fi = np.sum(dati,axis=0)
    L[1] = {(i,):Fi[i] for i in Omega if Fi[i] >= alpha*observation}
    k = 1
    while L[k]:
        L[k+1] = {}
        for s1 in L[k]:
            for s2 in L[k]:
                s3 = set(s1)-set(s2)
                if len(s3) == 1:
                    s12 = set(s1) | set(s2)
                    s12_is_good = True
                    for i in s12:
                        if tuple(sorted(s12-{i})) not in L[k]:
                            s12_is_good = False
                            break
                    if s12_is_good:
                        s12 = tuple(sorted(s12))
                        Fs12 = np.sum(np.all(dati[:,s12],axis=1))
                        if Fs12 >= alpha*observation:
                            L[k+1][s12]=Fs12
        k += 1
    return L

def get_rules(L, beta):
    R = []
    for k in L:
        for s in L[k]:
            for j in range(1,len(s)):
                for sub_s in combinations(set(s),j):
                    if L[len(sub_s)][tuple(sorted(sub_s))]*beta <= L[k][s]:
                        R.append([sub_s,tuple(set(s)-set(sub_s))])
    return R

L = get_frequent_itemsets(dati,alpha)

for k in L:
    for S in L[k]:
        print([items[i] for i in S])





R = get_rules(L,beta)
if R:
    print("association rules")
    for r in R:
        print([items[i] for i in r[0]]," -> ",[items[i] for i in r[1]])
else:
    print("no association rules")
    
    
############################  NEURAL NETWORK NORMALE
import pandas as pd
import numpy as np

dato=pd.read_csv("C:/Users/luigimissri/Desktop/FILE DA SALVARE/Data Science/Knoledge and Data Mining/Progetto/dati_anal.csv")
del dato["Unnamed: 0"]

trainx=dato.iloc[0:650,0:34]
trainy=dato.iloc[0:650,34:]
testx=dato.iloc[650:,0:34]
testy=dato.iloc[650:,34:]



import tensorflow as tf

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 650
display_step = 25

# Network Parameters
n_hidden_1 = 65 # 1st layer number of neurons
n_hidden_2 = 65# 2nd layer number of neurons
n_hidden_3 = 65
n_hidden_4 = 65
num_input = 34 
num_classes = 2 

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1], seed=123)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],seed=12)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], seed=1)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], seed=1234)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, num_classes],seed=23))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1],seed=50)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], seed= 75)),
    'b3': tf.Variable(tf.random_normal([n_hidden_3], seed= 13)),
    'b4': tf.Variable(tf.random_normal([n_hidden_4], seed= 44)),
    'out': tf.Variable(tf.random_normal([num_classes], seed=83))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Hidden fully connected layer with 256 neurons
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # Hidden fully connected layer with 256 neurons
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])    
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)


# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x = trainx
        batch_y = trainy
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
       
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: testx,
                                      Y: testy}))
    
############ FINO A QUI OK
    
import pandas as pd
import numpy as np

dat=pd.read_csv("C:/Users/luigimissri/Desktop/FILE DA SALVARE/Data Science/Knoledge and Data Mining/Progetto/det_factor.csv")
del dat["Unnamed: 0"]
dat.describe()
for nom in list(dat.columns):

    dat[nom] = dat[nom].astype('category')

dato=pd.get_dummies(dat)
del dato["num_1"]
del dato["num_2"]
del dato["num_3"]
del dato["num_4"]

num1=1-dato.iloc[:,34]
dato["num_1"]=num1

trainx=dato.iloc[0:650,0:34]
trainy=dato.iloc[0:650,34:]
testx=dato.iloc[650:,0:34]
testy=dato.iloc[650:,34:]


import tensorflow as tf

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 650
display_step = 25

# Network Parameters
n_hidden_1 = 65 # 1st layer number of neurons
n_hidden_2 = 65# 2nd layer number of neurons
n_hidden_3 = 65
n_hidden_4 = 65
num_input = 34 
num_classes = 2 

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1],seed=123)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],seed=12)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],seed=1)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4],seed=1234)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, num_classes],seed=23))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1],seed=50)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2],seed=75)),
    'b3': tf.Variable(tf.random_normal([n_hidden_3],seed=13)),
    'b4': tf.Variable(tf.random_normal([n_hidden_4],seed=44)),
    'out': tf.Variable(tf.random_normal([num_classes],seed=83))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Hidden fully connected layer with 256 neurons
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # Hidden fully connected layer with 256 neurons
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])    
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)


# Define loss and optimizer
#1- A+B
a=tf.minimum(1.0,1-prediction[:,0]+X[:,19])  
    
b=-tf.reduce_sum(a)

loss_o = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
loss_op=tf.add(loss_o,b)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op+100*b)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
sess=tf.Session() 

    # Run the initializer
sess.run(init)

for step in range(1, num_steps+1):
    batch_x = trainx
    batch_y = trainy
    # Run optimization op (backprop)
    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

    
    if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                             Y: batch_y})
        print("Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))
        
print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={X: testx,
                                  Y: testy}))
    
    

best = sess.run(prediction[:,0],feed_dict={X: testx, Y: testy})
   
#(testy.iloc[:,1]-best.round(2)).abs().sum()









    