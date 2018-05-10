import urllib2
from bs4 import BeautifulSoup


def weplay(x):
    # Weplay
    # Num de paginas 17 (hasta el momento)
    f = urllib2.urlopen("https://www.weplay.cl/juegos/juegosps4.html?dir=asc&order=price&p="+str(x))
    html = BeautifulSoup(f.read(), "html.parser")
    nombre = html.findAll("div",{"class":"containerNombreYMarca"})
    precio = html.findAll("div", {"class" : "price-box wePlayPrice"})
    for k in zip(nombre, precio):
        print k[0].find("h2",{"class":"product-name"}).getText()
        print k[1].find("span", {"class" : "price"}).getText()
        print "-----------------------------------------------------------------"

def Microplay():
    f = urllib2.urlopen("https://www.microplay.cl/productos/juegos?plataformas=playstation-4&sort=precio,asc")
    html = BeautifulSoup(f.read(), "html.parser")
    print html

#Microplay()

x = 260 >= ()

#for k in range(1,18):
#    weplay(k)