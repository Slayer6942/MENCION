import cv2, os

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


def escribir_datos(D, largo, tipo, archivo):
    for i in range(largo):
        string1 = str(D[i])
        archivo.write(string1[1: -1] + " , " + str(tipo) + " \n")


def main():
    datos = open("prueba.txt", "w")
    positivas = len(os.listdir("entrenamiento/positivas"))
    negativas = len(os.listdir("entrenamiento/negativas"))
    datosP = [0] * positivas
    datosN = [0] * negativas

    for nu in range(1, positivas):
        datosP[nu - 1] = analisis("entrenamiento/positivas/" + str(nu) + ".jpg", datos, "P")
    escribir_datos(datosP, positivas, 0, datos)
    for nu in range(1, negativas):
        datosN[nu - 1] = analisis("entrenamiento/negativas/" + str(nu) + ".jpg", datos, "N")
    escribir_datos(datosN, negativas, 0, datos)


if __name__ == '__main__':
    main()