import tensorflow as tf
import numpy as np
import cv2
from skimage import data, color, feature, io
import skimage.data

# Importar datos MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#HOG
HOG = []
for K in range(0, 59999):
	image_file = "PNG/training/" + str(K) + ".png" 
	img_bgr = cv2.imread(image_file)
	img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
	hog_vec = feature.hog(img_gray, orientations = 9, pixels_per_cell = (14, 14), cells_per_block = (1, 1), visualise = False)	
	HOG.append(hog_vec)	
	

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

#Establecer pesos modelo
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x, W) + b) 
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
init = tf.global_variables_initializer()

# Inicia el entrenamiento
with tf.Session() as sess:
	sess.run(init)
	print("Regresor logistico HOG\n")
    
	for epoch in range(25):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/100)
		
		for i in range(total_batch):
			HOG, t_y = mnist.train.next_batch(100)
			_, c = sess.run([optimizer, cost], feed_dict = {x: HOG, y: t_y})
			avg_cost += c / total_batch
			       
		if (epoch + 1) % 1 == 0:
			print("Epoca:", '%.2d' % (epoch+1), " Costo:", "{:.9f}".format(avg_cost))

    # Modelo de prueba
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calcular la precision
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("\nExactitud: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

