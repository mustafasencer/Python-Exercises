'''NUMPY TRAINING SESSION#-------------------------------------------
import numpy as np
a1 = np.array([[2,1],[3,4],[4,8]])
print (a1.ndim)
print (a1.itemsize)
a2 = np.array([[7,7],[3,5],[8,4]], dtype=np.float64)
print (a2)
print (a2.size)
print (a2.shape)
print (np.zeros((4,3)))
print (np.ones((4,3)))
print (np.linspace(1,5,10))
print (np.arange(1,5))
print (a2.reshape(2,3))
print (a1.reshape(3,2))
print (a2.ravel())
print ("bi break yap")
print (a2.min())
print (a2.sum())
print (a2.sum(axis=0))
print (np.sqrt(a2))
#print (a1.dot(a2))
print (a1[1])
print "Hade lan"
for cell in a1.flat:
    print (cell)
b1 = np.arange(6).reshape(3,2)
b2 = np.arange(6,12).reshape(3,2)
#print (b1.dot(b2))
print (np.vstack((b1,b2)))
print (np.hstack((b1,b2)))
print (np.hsplit(b1,2))
result = np.hsplit(b1,2)
print (result[1])
a = np.arange(12).reshape(3,4)
print (a)
b = a>4
print (b)
print (a[b])
a[b] = -1
print (a)
'''
#-------------------------------------------------------------------------------------
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_url(int):
    url = "http://charts.realclearpolitics.com/charts/{}.xml".format(str(int))
    xml = requests.get(url).text
    return xml

def get_color(xml):
    color_result ={}
    soup = BeautifulSoup(xml,"html.parser")
    for link in soup.findAll("graph"):
        color_result[link.attrs["title"]] = link.attrs["color"]
    return color_result
xml = get_url(1044)
print get_color(xml)
#___________________________________________________________________________________
def poll_data(xml):
    soup = BeautifulSoup(xml,"html.parser")
    liste = []
    result ={}
    for link in soup.find("series").findAll("value"):
        liste.append(str(link.get_text()))
    liste = pd.to_datetime(liste)
    liste1 = [float(i.get_text()) if i.get_text() else np.nan for i in soup.find("graph",{"title":"Approve"}).findAll("value")]
    liste2 = [float(y.get_text()) if y.get_text() else np.nan for y in soup.find("graph",{"title":"Disapprove"}).findAll("value") ]
    result["Approve"] = liste1
    result["Disapprove"] = liste2
    result["date"] = liste
    result = pd.DataFrame(result)
    return result

def poll_plot(int):
    xml = get_url(int)
    color = get_color(xml)
    result = poll_data(xml)
    norm = result[color.keys()].sum(axis=1)/100
    for c in color.keys():
        result[c] /= norm
    for label,color in color.items():
        plt.plot(result["date"],result[label],color=color,label=label)
    plt.xlabel("Date")
    plt.ylabel("Poll Date/Normalized")
    plt.legend(loc="best")
    plt.xticks(rotation=30)


#________________________________________________________________________________

#______________________________________________________________________________

