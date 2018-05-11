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

x = 1
while x <= 12:
    imagen = Image.open("images/Imagen_"+str(x)+".jpg")
    doc = etree.parse('images/imagen_'+str(x)+'.xml')
    cont =1

    raiz=doc.getroot()
    #for i in range(0,5):
    for i in range(4,len(raiz),1):
        t1 = raiz[i]
        #------------------------------------------------------------
        # Esta seccion se va adentrando nivel, por nivel en el XML, hasta llegar al nivel de las posiciones
        t2 = t1[9]
        p1 = t2[1]
        p2 = t2[2]
        p3 = t2[3]
        p4 = t2[4]
        #------------------------------------------------------------
        X1 = int(p1[0].text)
        Y1 = int(p1[1].text)
        X2 = int(p2[0].text)
        Y2 = int(p2[1].text)
        X3 = int(p3[0].text)
        Y3 = int(p3[1].text)
        X4 = int(p4[0].text)
        Y4 = int(p4[1].text)
        dibujo = ImageDraw.Draw(imagen)
        px1, py1,px2,py2 = comprobar(X1,Y1,X2,Y3)
        if px1 < px2:
            dimX = px2 - px1
        else:
            dimX = px1 - px2
        if py1 < py2:
            dimY = py2 - py1
        else:
            dimY = py1 - py2
        #print px1,py1,dimX,dimY
        #Obtenemos tamanios minimos y maximo de X e Y
        maxX,maxY, minX, minY = max_min(dimX, maxX, minX, dimY, maxY, minY)

        region = imagen.crop((px1,py1,px2,py2))



        region.save("entrenamiento/positivas/"+str(x)+"_"+str(cont)+".jpg") #El guardado de imagenes ya no es necesario
        arch.write("..\..\images\Imagen_"+str(x)+".jpg 1 "+ str(px1)+" "+str(py1)+" "+str(dimX)+" "+str(dimY)+"\n")
        arch_etiqueta.write(str(cont)+","+str(px1)+","+str(py1)+","+str(dimX)+","+str(dimY)+"\n")
        cont += 1
        #cont2=1

        reg1 = imagen.crop((px1, py1+1, px2, py2))
        reg1.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) +"_1"+ ".jpg")  # El guardado de imagenes ya no es necesario

        reg2 = imagen.crop((px1+1, py1+1, px2, py2))
        reg2.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) + "_2" +  ".jpg")  # El guardado de imagenes ya no es necesario

        reg3 = imagen.crop((px1+1 , py1, px2, py2))
        reg3.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) + "_3" +   ".jpg")  # El guardado de imagenes ya no es necesario

        reg4 = imagen.crop((px1+1, py1-1, px2, py2))
        reg4.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) + "_4" +  ".jpg")  # El guardado de imagenes ya no es necesario

        reg5 = imagen.crop((px1, py1-1, px2, py2))
        reg5.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) + "_5" + ".jpg")  # El guardado de imagenes ya no es necesario

        reg6 = imagen.crop((px1-1 , py1-1, px2, py2))
        reg6.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) + "_6" + ".jpg")  # El guardado de imagenes ya no es necesario

        reg7 = imagen.crop((px1-1, py1, px2, py2))
        reg7.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) + "_7" +  ".jpg")  # El guardado de imagenes ya no es necesario

        reg8 = imagen.crop((px1 -1, py1+1, px2, py2))
        reg8.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) + "_8" + ".jpg")  # El guardado de imagenes ya no es necesario
#-----------------------------
        reg9 = imagen.crop((px1, py1 + 2, px2, py2))
        reg9.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) + "_9" + ".jpg")  # El guardado de imagenes ya no es necesario

        reg10 = imagen.crop((px1 + 2, py1 + 2, px2, py2))
        reg10.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) + "_10" +".jpg")  # El guardado de imagenes ya no es necesario

        reg11 = imagen.crop((px1 + 2, py1, px2, py2))
        reg11.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) + "_11" + ".jpg")  # El guardado de imagenes ya no es necesario

        reg12= imagen.crop((px1 + 2, py1 - 2, px2, py2))
        reg12.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) + "_12" +  ".jpg")  # El guardado de imagenes ya no es necesario

        reg13 = imagen.crop((px1, py1 - 2, px2, py2))
        reg13.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) + "_13" +  ".jpg")  # El guardado de imagenes ya no es necesario

        reg14 = imagen.crop((px1 - 2, py1 - 2, px2, py2))
        reg14.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) + "_14" +  ".jpg")  # El guardado de imagenes ya no es necesario

        reg15= imagen.crop((px1 - 2, py1, px2, py2))
        reg15.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) + "_15" + ".jpg")  # El guardado de imagenes ya no es necesario

        reg16= imagen.crop((px1 - 2, py1 + 2, px2, py2))
        reg16.save("entrenamiento/positivas/" + str(x) + "_" + str(cont) + "_16" +  ".jpg")  # El guardado de imagenes ya no es necesario

        #arch.write("..\..\images\Imagen_" + str(x) + ".jpg 1 " + str(px1) + " " + str(py1) + " " + str(dimX) + " " + str(dimY) + "\n")
        #arch_etiqueta.write(str(cont2) + "," + str(px1) + "," + str(py1) + "," + str(dimX) + "," + str(dimY) + "\n")
        #cont2=cont2+1

    x+= 1
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
