import requests
import pandas as pd
import xml.etree.ElementTree as et
import re
import matplotlib.pyplot as plt

def get_poll(int):
    xml = requests.get("http://charts.realclearpolitics.com/charts/"+str(int)+".xml").text
    return xml
def _strip(s):
    return re.sub(r'[\W_]+', '', s)
def plot_colors(xml):
    tree = et.fromstring(xml)
    result ={}
    key1 = tree.find("./graphs/graph[@gid='1']").get("title")
    key2 = tree.find("./graphs/graph[@gid='2']").get("title")
    result[key1] = tree.find("./graphs/graph[@gid='1']").get("color")
    result[key2] = tree.find("./graphs/graph[@gid='2']").get("color")
    return result

def rcp_poll_data(xml):
    tree = et.fromstring(xml)
    dates = [] ; graphlist1 = [] ; graphlist2 = []
    g1_title = tree.find("./graphs/graph[@gid='1']").get("title")
    g2_title = tree.find("./graphs/graph[@gid='2']").get("title")

    for s,g1,g2 in zip(tree.iterfind("./series/value"),tree.iterfind(".graphs/graph[@gid='1']/value"),tree.iterfind(".graphs/graph[@gid='2']/value")):
        dates.append(s.text)
        graphlist1.append(g1.text)
        graphlist2.append(g2.text)
    dates = pd.to_datetime(dates)
    result = pd.DataFrame({"date":dates,g1_title:graphlist1,g2_title:graphlist2})
    return result[[g1_title]].sum(axis=1)
xml = get_poll(1171)
print rcp_poll_data(xml)


'''def poll_plot(int):
    xml = get_poll(int)
    data = rcp_poll_data(xml)
    color = plot_colors(xml)

    #data = data.rename(columns={c: _strip(c) for c in data.columns})

    norm = data[color.keys()].sum(axis=1)/100
    for c in color.keys():
        data[c] /=norm

    for label,colora in color.items():
        plt.plot(data["date"],data[label], color=colora,label=label)
'''

