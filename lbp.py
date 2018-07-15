import cv2, os, csv
from heapq import merge
from skimage import exposure
from skimage import feature
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure


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
    val_ar.append(get_pixel(img, center, x - 1, y + 1))  # top_right
    val_ar.append(get_pixel(img, center, x, y + 1))  # right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))  # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y))  # bottom
    val_ar.append(get_pixel(img, center, x + 1, y - 1))  # bottom_left
    val_ar.append(get_pixel(img, center, x, y - 1))  # left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))  # top_left
    val_ar.append(get_pixel(img, center, x - 1, y))  # top

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val

def analisis(imagen, archivo, tipo):
    img_bgr = cv2.imread(imagen)
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    arry = [0] * (height * width)
    cont = 0
    for i in range(0, height):
        for j in range(0, width):
            arry[cont] = lbp_calculated_pixel(img_gray, i, j)
            cont += 1
    return arry

def HOG(imagen):
    (H, hogImage) = feature.hog(imagen, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm = "L1",visualize = True)
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")

    cv2.imshow("HOG Image", hogImage)

def escribir_datos(D, largo, tipo, archivo, maximo):
    maximo = maximo
    for i in range(largo):
        diferencia = maximo - len(D[i])
        agregar = D[i] + ([0]*diferencia)
        string1 = str(agregar)
        archivo.write(string1[1: -1] + "," + str(tipo) + "\n")
        #X= (X - mean(X))/std(X)

def calcular_maximo(datos1,datos2):
    maximo = len(datos1[0])
    for largo in datos1:
        if len(largo) > maximo:
            maximo = len(largo)
    for largo in datos2:
        if len(largo) > maximo:
            maximo = len(largo)
    return maximo

def main():
    datos = open("lbp.csv", "w")
    positivas = len(os.listdir("entrenamiento/positivas"))-1
    negativas = len(os.listdir("entrenamiento/negativas"))-1
    #positivas = 10
    #negativas = 10
    datosP = [0] * (positivas)
    datosN = [0] * (negativas)

    for nu in range(positivas):
        datosP[nu] = analisis("entrenamiento/positivas/" + str(nu) + ".jpg", datos, "P")
    for nu in range(negativas):
        datosN[nu] = analisis("entrenamiento/negativas/" + str(nu) + ".jpg", datos, "N")
    maximo = calcular_maximo(datosP, datosN)
    escribir_datos(datosP, positivas, 1, datos, maximo)
    escribir_datos(datosN, negativas, 0, datos, maximo)

if __name__ == '__main__':
    main()