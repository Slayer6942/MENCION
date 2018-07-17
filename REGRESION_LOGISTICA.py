from __future__ import print_function
import tensorflow as tf
import numpy as np

# Importar datos MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# tf entrada de graficos
x = tf.placeholder(tf.float32, [None, 784]) # Imagen de datos de mnist de forma 28 * 28 = 784
y = tf.placeholder(tf.float32, [None, 10]) # Reconocimiento de 0-9 dígitos => 10 clases

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
    
    print("Regresor logistico pixel\n")	
    
    for epoch in range(25):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/100)
	
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            # Ejecuta la optimización op (backprop) y el costo op (para obtener el valor de pérdida)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            # Calcula la pérdida promedio
            avg_cost += c / total_batch
            
        if (epoch + 1) % 1 == 0:
            print("Epoca:", '%.2d' % (epoch+1), " Costo:", "{:.9f}".format(avg_cost))

    # Modelo de prueba
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calcular la precision
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("\nExactitud: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
