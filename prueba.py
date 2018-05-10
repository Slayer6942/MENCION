import urllib2
from bs4 import BeautifulSoup

f = urllib2.urlopen("https://www.zmart.cl/JuegosPS4")
html = BeautifulSoup(f.read(), "html.parser")
html = html.find("input",{"id":"curPage_32641"}).get("13")
print html
