from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from PIL import Image

#Importa datos desde MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
  
# tf Graph input
x = tf.placeholder(tf.float32, shape=[None, 784]) #Imagenes
y = tf.placeholder(tf.float32, shape=[None, 10])  #Etiquetas

#Capa oculta
W_1 = tf.Variable(tf.truncated_normal(shape = [784,512], stddev = 0.2))
b_1 = tf.Variable(tf.zeros([512]))
#Capa salida
W_2 = tf.Variable(tf.truncated_normal(shape = [512,10], stddev = 0.2))
b_2 = tf.Variable(tf.zeros([10]))

#Arquitectura de la red neuronal
def NN(x):
    #Capa oculta 
	z_1 = tf.matmul(x,W_1) + b_1 #Combinación lineal
	a_1  = tf.nn.relu(z_1)       #Activación 
    #Capa salida
	z_2 = tf.matmul(a_1,W_2) + b_2 #Combinación lineal
    
	return z_2

#Funcion de costo
y_ = NN(x)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_, labels=y))

#Predicciones
train_pred = tf.nn.softmax(y_) #Prediccion conjunto entrenamiento
y_valid = NN(mnist.validation.images)
valid_pred = tf.nn.softmax(y_valid) #Prediccion conjunto validacion

#Optimizador
opt = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Sesion e inicializacion de variables
sess = tf.Session() 
sess.run(tf.global_variables_initializer())
print("Red neuronal pixel\n")

#Precisión
def precision(predicciones, etiquetas):
    return (100.0 * np.sum(np.argmax(predicciones, 1) == np.argmax(etiquetas, 1))
          / predicciones.shape[0])
 
#Entrenamiento 
pasos = 5000
for i in range(pasos):
    batch = mnist.train.next_batch(100)
    _,costo,predicciones =  sess.run([opt, cross_entropy, train_pred],  feed_dict = {x: batch[0], y: batch[1]})
    
    if (i % 200 == 0):
       print("Paso %.4d:  Costo: %f" % (i, costo))
    
#Prueba     
y_test = NN(mnist.test.images)
test_prediction = tf.nn.softmax(y_test)
print("\nPRECISION PRUEBA: %.1f%%" % precision(test_prediction.eval(session = sess), mnist.test.labels))
