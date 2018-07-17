from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import numpy as np
from skimage import data, color, feature, io
import skimage.data
import random

#Importa datos desde MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] <= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):

    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


for K in range(0, 59999):
	image_file = "PNG/training/" + str(K) + ".png" 
	img_bgr = cv2.imread(image_file)
	height, width, channel = img_bgr.shape
	img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
	LBP = [0] * (height * width)
	cont = 0
		
	for i in range(0, height):
		for j in range(0, width):
			LBP[cont] = lbp_calculated_pixel(img_gray, i, j)
			cont += 1		
  
# tf Graph input
x = tf.placeholder(tf.float32, shape=[None, 784]) #Imagenes
y = tf.placeholder(tf.float32, shape=[None, 10])  #Etiquetas

#Establecer pesos modelo
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construccion del modelo
pred = tf.nn.softmax(tf.matmul(x, W) + b) #Función exponencial normalizada

#Funcion de costo - Minimiza el error usando entropía cruzada
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

#Optimizador - Descenso de gradiente
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

# Inicialice las variables (es decir, asigne su valor predeterminado)
init = tf.global_variables_initializer()

# Inicia el entrenamiento
with tf.Session() as sess:

    sess.run(init)
    
    print("Regresor logistico LBP\n")	
    
    for epoch in range(25):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/100)
	
        for i in range(total_batch):            
            LBP, batch_ys = mnist.train.next_batch(100)
            # Ejecuta la optimización op (backprop) y el costo op (para obtener el valor de pérdida)
            _, c = sess.run([optimizer, cost], feed_dict={x: LBP, y: batch_ys})
            # Calcula la pérdida promedio
            avg_cost += c / total_batch
            
        if (epoch + 1) % 1 == 0:
            print("Epoca:", '%.2d' % (epoch+1), " Costo:", "{:.9f}".format(avg_cost))

    # Modelo de prueba
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calcular la precision
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("\nExactitud: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

