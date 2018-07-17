from PIL import Image

def crop(img):
    width, heigth = img.size
    #redimensiono el espacio que detecta un numero
    A = 0
    cont = 0
    #alto de la ventana
    endY = heigth


    #### ITERACIONES x IMAGEN deben ser 5
    for i in range(5):
        #inicio en el punto 0,0
        startY = (heigth - heigth)+A
        #promedio de medida para los numeros, en cada iteracion redimensiono
        startX = ((width / 10))-A

        #### SUBDIVISIONES rango de 10 por ventana (contando el - )
        for j in range(10):
            area = (startY, 0, startX, endY)
            cropped_img = img.crop(area)
            cropped_img.save("entrenamiento/subdivisiones/"+str(cont) + ".jpg")
            #cropped_img.show()
            startX += 28-A
            startY += 28-A
            cont+=1
        A+=1


if __name__ == "__main__":
    img = Image.open("entrenamiento/Imagen_8.jpg")
    crop(img)