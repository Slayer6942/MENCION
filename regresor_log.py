from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import scipy as sp
from numpy import *

data= sp.matrix(sp.loadtxt("lbp.txt", delimiter=','))
x = data[:,0:-1]
x= (x - mean(x))/std(x)
y= data[:,-1]


kfold = model_selection.ShuffleSplit(n_splits= 5, test_size= 0.2, train_size=0.8)
model = LogisticRegression()
result = model_selection.cross_val_score(model, x,y, cv = kfold)

print("Porcentajes por test: ")
for i in result:
    print(" (0: .2f".format(i * 100.0)+ "%")
    print("desviacion estandar: %.3f%% " % (result.std()*100.0))