from PIL import Image
import random

def obtener_pos():
    archi = open("pos_imagenes.txt","r")
    datos = []
    for k in archi:
        m = k.rstrip('\n')
        datos.append(m)
    archi.close()
    return datos

def negToPos(x):
    if x < 0:
        x = x * -1
    return x

imagen = Image.open("images/Imagen_1.jpg")

dimX = [177,240,303,366,430]
dimY = [106,27, 66]

arch = open("entrenamiento/negativas/negativas.txt","w")
cont = 0

datos_pos = obtener_pos()
while cont <= 1000:
    RDX = dimX[random.randint(0,len(dimX)-1)]
    RDY = dimY[random.randint(0,len(dimY)-1)]

    xR1 = random.randint(0, 500 - RDX)
    xR2 = random.randint(1000, imagen.size[0] - RDX)
    RanDx = random.randint(1, 2)
    y = random.randint(0,imagen.size[1]-RDY)

    if RanDx == 1:
        x = xR1
    else:
        x = xR2

    caja = (x,y,x+RDX,y+RDY)

    region = imagen.crop(caja)

    region.save("entrenamiento/negativas/"+str(cont)+".jpg")
    arch.write(str(cont)+".jpg\n")
    cont+=1
arch.close()
