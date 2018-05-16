import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from numpy import mean, std

degree = 6
lambda_ = 0.001


# def map_feature(X1, X2):
#  degree = 40
#  out = sp.matrix([
#          list(map(lambda c: c.real, sp.multiply(sp.power(X1, (i-j)), sp.power(X2, j)).flatten().tolist()))
#            for i in range(0, degree+1)
#            for j in range(0, i+1)
#        ]).T
#  return out

def mapFeature(X1, X2):
    # degree = 20
    if isinstance(X1, float):
        check = sp.ones((1, 1))
        X1 = X1 * check
        X2 = X2 * check

    m, n = X1.shape
    dg = (degree + 1) * (degree + 2) / 2
    out = sp.ones((m, dg))
    k = 0
    for i in range(degree + 1):
        for j in range(i + 1):
            aux3 = np.multiply(np.power(X1, i - j), np.power(X2, j))
            aux3 = np.asmatrix(aux3)
            out[:, k] = aux3.getA1()
            k = k + 1
    return out


# Define sigmoid, cost function and gradients
def sigmoid(z):
    return 1 / (1 + sp.exp(-z))


def cost_function(theta, X, Y):
    m, n = X.shape
    #  lambda_=0.001000
    theta = sp.matrix(theta).T
    J = (1 / m) * (-Y.T * sp.log(sigmoid(X * theta)) - ((1 - Y).T * sp.log(1 - sigmoid(X * theta))))
    regul = np.multiply(theta, theta)
    aux1 = sum(regul[1:m])
    J = J + (lambda_ / m) * aux1
    return J[0, 0]


def gradients(theta, X, Y):
    m, n = X.shape
    #  lambda_=0.00100
    theta = sp.matrix(theta).T
    grad = ((1 / m) * X.T * (sigmoid(X * theta) - Y)).T
    grad = sp.squeeze(sp.asarray(grad))
    aux = theta[1:n]
    grad[1:n] = grad[1:n] + (lambda_ / m) * aux.getA1()
    return grad


def predict(theta, X):
    return sp.around(sigmoid(X * theta))


# Load data from data source 1
data = sp.matrix(sp.loadtxt("ex2data2.txt", delimiter=','))
X = data[:, 0:2]
oldX = X

X = mapFeature(oldX[:, 0], oldX[:, 1])
# print(X.shape)
# X = (X - mean(X))/std(X)
Y = data[:, 2]
m, n = X.shape
# print("size")
# print(X.shape)
# Compute cost and gradients
# Initialize
# X = sp.hstack((sp.ones((m, 1)), X))

theta = sp.zeros(n)  # Use row vector instead of column vector for applying optimization

# Optimize using fmin_bfgs
res = fmin_bfgs(cost_function, theta, fprime=gradients, disp=True, args=(X, Y))
theta = sp.matrix(res).T

# X=oldX
# Plot fiqure 1 (data)
plt.clf()
plt.figure(1)
plt.xlabel('x1')
plt.ylabel('x2')

pos = sp.where(Y == 1)[0]
neg = sp.where(Y == 0)[0]

plt.plot(X[pos, 1], X[pos, 2], 'k+', linewidth=2, markersize=7)
plt.plot(X[neg, 1], X[neg, 2], 'ko', markerfacecolor='y', markersize=7)

# Plot fiqure 2 (decision boundary)
plt.figure(2)
plt.xlabel('x1')
plt.ylabel('x2')

pos = sp.where(Y == 1)[0]
neg = sp.where(Y == 0)[0]

plt.plot(X[pos, 1], X[pos, 2], 'ko', markerfacecolor='b', linewidth=2, markersize=10)
plt.plot(X[neg, 1], X[neg, 2], 'ko', markerfacecolor='y', markersize=10)

p = predict(theta, X)
r = sp.mean(sp.double(p == Y)) * 100

plt.plot(X[pos, 1], X[pos, 2], 'k+', linewidth=2, markersize=7)
plt.plot(X[neg, 1], X[neg, 2], 'ko', markerfacecolor='y', markersize=7)

u = sp.linspace(-1, 1.5, 50)
v = sp.linspace(-1, 1.5, 50)
z = sp.matrix(
    sp.reshape(
        [mapFeature(u[i], v[j]) * theta
         for i in range(0, len(u))
         for j in range(0, len(v))
         ], [50, 50])).T

plt.contour(u, v, z, 0, linewidth=2)

print("Train Accuracy: {r}%".format(**locals()))