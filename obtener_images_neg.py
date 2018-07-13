from PIL import Image
import random

#Esta funcion trandorma as posisiones negativas a positivas para que sea mas facil tratarlas
def negToPos(x):
    if x < 0:
        x = x * -1
    return x

imagen = Image.open("images/Imagen_1.jpg")

#Estoy arrays guardan las 5 dimenciones en X y las 3 dimenciones en Y
dimX = [177,240,303,366,430]
dimY = [106,27, 66]

arch = open("entrenamiento/negativas/negativas.txt","w")
cont = 1

#En este ciclo generamos las imagenes negativas
while cont <= 10000:
    #Seleccionamos un X,Y aleatorio de los array dimX y dimY
    RDX = dimX[random.randint(0,len(dimX)-1)]
    RDY = dimY[random.randint(0,len(dimY)-1)]

    #Aqui obtenemos una posicion entre 0 y 500, 1000 y el ancho de la imagen
    #Esto significa que eliminamos la seccion de la imagen en la cual se encuentran los rut
    #Y poder generar las imagenes negativas con secciones inferrioes al 50% del rut
    xR1 = random.randint(0, 500 - RDX)
    xR2 = random.randint(1000, imagen.size[0] - RDX)
    RanDx = random.randint(1, 2)
    y = random.randint(0,imagen.size[1]-RDY)

    #Esta condicion nos deja seleccionar ambos sectores de la imagen en la cual podemos obtener imagenes
    #Si el valor es igual a 1 el random en X es entre 0 y 5, de lo contrario el X sera entre 1000 y el ancho de la imagen
    if RanDx == 1:
        x = xR1
    else:
        x = xR2

    #Creamos la seccion que recortaremos y luego con la funcion "crop" recortamos dicha seccion
    caja = (x,y,x+RDX,y+RDY)
    region = imagen.crop(caja)

    #Guardamos el recorte realizado
    region.save("entrenamiento/negativas/"+str(cont)+".jpg")
    arch.write(str(cont)+".jpg\n")
    cont+=1
arch.close()
