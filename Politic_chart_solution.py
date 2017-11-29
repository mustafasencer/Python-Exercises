import pandas as pd
from pattern import web
import requests
import numpy as np
import re
import matplotlib.pyplot as plt
def get_poll_xml(poll_id):
    url = "http://charts.realclearpolitics.com/charts/%i.xml" % int(poll_id)
    return requests.get(url).text
def _strip(s):
    return re.sub(r'[\W_]+', '', s)
def plot_colors(xml):

    dom = web.Element(xml)
    result = {}
    for graph in dom.by_tag('graph'):
        title = _strip(graph.attributes['title'])
        result[title] = graph.attributes['color']
    return result
def rcp_poll_data(xml):
    '''dom = web.Element(xml)
    result = {}

    dates = dom.by_tag('series')[0]
    dates = {n.attributes['xid']: str(n.content) for n in dates.by_tag('value')}

    keys = dates.keys()

    result['date'] = pd.to_datetime([dates[k] for k in keys])

    for graph in dom.by_tag('graph'):
        name = graph.attributes['title']
        data = {n.attributes['xid']: float(n.content)
                if n.content else np.nan for n in graph.by_tag('value')}
        result[name] = [data[k] for k in keys]

    result = pd.DataFrame(result)
    result = result.sort_values("date",ascending=True)
    result = result.reset_index(drop=True)
    return result
xml = get_poll_xml(1044)
print rcp_poll_data(xml)'''
    result = {}
    dom = web.Element(xml)
    dates = dom.by_tag("series")[0]
    '''dates = {n.attributes["xid"]:str(n.content) for n in dates.by_tag("value")}
    result["date"] = pd.to_datetime([dates[k] for k in dates.keys()])
    for value in dom.by_tag("graph"):
        name = value.attributes['title']
        dic = {n.attributes["xid"]:float(n.content) if n.content else np.nan for n in value.by_tag("value")}
        result[name] = [dic[k] for k in dic.keys()]
    result = pd.DataFrame(result)
    result = result.sort_values("date",ascending=True)
    result = result.reset_index(drop=True)'''
    return dates
xml = get_poll_xml(1044)
print rcp_poll_data(xml)

def poll_plot(poll_id):
    xml = get_poll_xml(poll_id)
    data = rcp_poll_data(xml)
    colors = plot_colors(xml)
    data = data.rename(columns = {c: _strip(c) for c in data.columns})
    norm = data[colors.keys()].sum(axis=1) / 100
    for c in colors.keys():
        data[c] /= norm
    for label, color in colors.items():
        plt.plot(data.date, data[label], color=color, label=label)
    plt.xticks(rotation=30)
    plt.legend(loc='best')
    plt.xlabel("Date")
    plt.ylabel("Normalized Poll Percentage")
