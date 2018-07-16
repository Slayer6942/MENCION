from PIL import Image, ImageDraw
from lxml import etree

#Esta funcion invierte las posiciones de los vertices de las imagenes, para que todos los datos comiences
#desde la esquina superior izquierda
def comprobar(x1,y1,x2,y2):
    X1=x1
    Y1=y1
    X2=x2
    Y2=y2
    if x1 > x2:
        X1 = x2
        X2= x1
    if y1 > y2:
        Y1 = y2
        Y2 = y1
    return X1,Y1,X2,Y2

#Calculamos los valores maximos y minimos de las dimenciones de las imagenes de los ejes X y Y
def max_min(x, maxX, minX, y, maxY, minY):
    if maxX <= x:
        maxX = x
    elif minX >= x or minX == 0:
        minX = x

    if maxY <= y:
        maxY = y
    elif minY >= y or minY == 0:
        minY = y

    return maxX, maxY, minX, minY

arch = open("entrenamiento/positivas/positivas.txt","w")
arch_etiqueta = open("etiquetas_imagenes.txt","w")
promedio, maxX, maxY, minX, minY = 0, 0, 0, 0, 0

#esta variable contiene la cantidad de pixeles que deben correrse cada imagen en X,Y para ovtener las 8 variaciones
variaciones = [[-2,-2],[-2,0],[-2,2],[0,2],[2,2],[2,0],[2,-2],[0,-2],[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[0,0]]
nombre = 0
x = 1

while x <= 1:
    imagen = Image.open("images/Imagen_"+str(x)+".jpg")
    doc = etree.parse('images/imagen_'+str(x)+'.xml')
    cont =1


    raiz=doc.getroot()
    for i in range(4,len(raiz),1):
        t1 = raiz[i]

        # Esta seccion se va adentrando nivel, por nivel en el XML, hasta llegar al nivel de las posiciones
        t2 = t1[9]
        p1 = t2[1]
        p2 = t2[2]
        p3 = t2[3]
        p4 = t2[4]

        #Accedo a las etiquetas que contienen lsa posiciones de cada bounding-box
        X1 = int(p1[0].text)
        Y1 = int(p1[1].text)
        X2 = int(p2[0].text)
        Y2 = int(p2[1].text)
        X3 = int(p3[0].text)
        Y3 = int(p3[1].text)
        X4 = int(p4[0].text)
        Y4 = int(p4[1].text)

        dibujo = ImageDraw.Draw(imagen)

        #Se realiza un cambio de variables en caso de que el bounding-box comienze desde abajo y no desde el primer punto superrior derecho
        px1, py1,px2,py2 = comprobar(X1,Y1,X2,Y3)
        if px1 < px2:
            dimX = px2 - px1
        else:
            dimX = px1 - px2
        if py1 < py2:
            dimY = py2 - py1
        else:
            dimY = py1 - py2

        #Obtenemos tamanios minimos y maximo de X e Y
        maxX,maxY, minX, minY = max_min(dimX, maxX, minX, dimY, maxY, minY)

        cont2=1

        for col in variaciones:
            zona = imagen.crop((px1+col[0], py1+col[1], px2, py2))
            zona.save("entrenamiento/positivas/" + str(nombre)+ ".jpg")
            arch.write("..\..\images\Imagen_" + str(x) + ".jpg 1 " + str(px1+col[0]) + " " + str(py1+col[1]) + " " + str(dimX) + " " + str(dimY) + "\n")
            arch_etiqueta.write(str(cont) + "," + str(px1+col[0]) + "," + str(py1+col[1]) + "," + str(dimX) + "," + str(dimY) + "\n")
            cont += 1
            cont2 += 1
            nombre += 1

    x+= 1   #No mover, ni identar hacia ningun lado

arch.close()
arch_etiqueta.close()

promedio = (maxY + minY) / 2
tam1 = minX + (maxX-minX)/4
tam2 = tam1 + (maxX-minX)/4
tam3 = tam2 + (maxX-minX)/4
print("3 Tamanios en Y")
print("Maximo:", maxY)
print("Minimo:", minY)
print("Promedio: ", promedio)
print("\n")
print("5 Tamanios en X")
print("Minimo:", minX)
print("Tamanios intermedios: ", tam1, tam2, tam3)
print("Maximo:", maxX)
