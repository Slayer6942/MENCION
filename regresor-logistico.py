import pandas as pd
import scipy as sp
import numpy as np
from numpy import *
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Creamos el Modelo de Regresion Logistica
#Creamos data frame de los datos
dataframe = pd.read_csv("lbp.csv")
X = np.array(dataframe.drop(['clase'],1))
y = np.array(dataframe['clase'])
X = (X - mean(X))/std(X)
print(dataframe.groupby('clase').size())

#Creamos nuestro modelo y hacemos que se ajuste (fit) a nuestro conjunto de entradas X y salidas "y"
model = linear_model.LogisticRegression()
model.fit(X,y)

#Validacion de nuestro modelo
#Subdividimos nuestros datos de entrada en forma aleatoria (mezclados) utilizando 80% de registros para entrenamiento y 20% para validar.
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

#Compilamos nuestro modelo de Regresion Logistica
name='Logistic Regression' ; name2 = 'Presicion' ; name3 = 'Desviacion estandar'
kfold = model_selection.KFold(n_splits=5, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s:\n %s: %f \n %s:(%f)" % (name, name2, cv_results.mean()*100, name3, cv_results.std()*100)
print(msg)

#Validamos el modelo con datos nuevos (cross validation set)
predictions = model.predict(X_validation)
print("Precision en datos de validacion cruzada:", accuracy_score(Y_validation, predictions)*100)
print(classification_report(Y_validation, predictions))

#Creamos data frame para probar las predicciones de nuestro modelo
dataframe_prueba = pd.read_csv("datos-prueba.csv")
pX = np.array(dataframe_prueba.drop(['clase'],1))
pX = (pX - mean(X))/std(pX)
yNew = model.predict(pX) #Me entrega las predicciones redondeadas

#en este vector se guardaran los datos de prediccion
yNew = model.predict_proba(pX) #Me entrega las predicciones en porcentaje

# Muestro entradas y probabilidades de prediccion
mejor_score = [[0,0]]
for i in range(len(pX)):
    #string.append(yNew[i][1])
    mejor_score.sort()
    mejor_score.extend([[yNew[i][1],i]])
    if len(mejor_score) > 200:
        mejor_score.pop(0)

    print("Img %s\n %s \nPrediccion = %s \n" % (i,pX[i], yNew[i]))

mejor_score.sort(reverse = True)
arch = open('fileScore.txt',"w")
for k in mejor_score:
    arch.write(str(k[0])+", "+str(k[1])+"\n")