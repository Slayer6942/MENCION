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

dimX = [50,100,150,200,250]
dinY = [50,100, 150]

arch = open("entrenamiento/negativas/negativas.txt","w")
cont = 0

datos_pos = obtener_pos()
while cont <= 100:
    RDX = dimX[random.randint(0,len(dimX)-1)]
    RDY = dimX[random.randint(0,len(dinY)-1)]
    x = random.randint(0,imagen.size[0]-RDX)
    y = random.randint(0,imagen.size[1]-RDY)

    for data in datos_pos:
        data = data.split(",")
        mitad = int(data[0]) + (int(data[2]) / 2)
        print data#, mitad, x,y
        if x > int(data[0]) and x < mitad and y > int(data[1]) and y < int(data[1])+int(data[3]):
            x = x+(int(data[2]) / 2)+100
            print(cont)
        if x < int(data[0]) and x>500:
            x = random.randint(0,400)
         

    caja = (x,y,x+RDX,y+RDY)

    region = imagen.crop(caja)

    region.save("entrenamiento/negativas/"+str(cont)+".jpg")
    arch.write(str(cont)+".jpg\n")
    cont+=1
arch.close()
