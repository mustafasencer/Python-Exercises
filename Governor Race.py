import requests
from bs4 import BeautifulSoup
from fnmatch import fnmatch
from pattern import web
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
def get_poll_xml(poll_id):
    url = "http://charts.realclearpolitics.com/charts/%i.xml" % int(poll_id)
    return requests.get(url).text
def _strip(s):
    return re.sub(r'[\W_]+', '', s)
def plot_colors(xml):
    dom = web.Element(xml)
    result = {}
    general = dom.by_tag("div#polling-data-full")[0]
    if_name = general.by_tag("th")
    name_1 = str(if_name[3].content.split("(")[0].strip())
    name_2 = str(if_name[4].content.split("(")[0].strip())
    if "R" in str(if_name[3].content).split("(")[1]:
        result[name_1] = "#D30015"
        result[name_2] = "#3B5998"
    else:
        result[name_1] = "#3B5998"
        result[name_2] = "#D30015"
    if len(if_name)>6:
        name_3 = str(if_name[5].content).split("(")[0].strip()
        a = str(if_name[3].content).split("(")[1]
        b = str(if_name[4].content).split("(")[1]
        if "R" in a:
            result[name_1] = "#D30015"
            if "D" in b:
                result[name_2] = "#3B5998"
                result[name_3] = "#01DF01"
            else:
                result[name_2] = "#01DF01"
                result[name_3] = "#3B5998"
        elif "D" in a:
            result[name_1] = "#3B5998"
            result[name_2] = "#D30015"
            result[name_3] = "#01DF01"
        elif "I" in a:
            if "R" in b:
                result[name_1] = "#01DF01"
                result[name_2] = "#D30015"
                result[name_3] = "#3B5998"
            else:
                result[name_1] = "#01DF01"
                result[name_2] = "#3B5998"
                result[name_3] = "#D30015"
    return result
def is_gov_race(l):
    pattern = "/epolls/????/governor/??/*-*.html"
    return fnmatch(l,pattern)
def find_governor_races(html):
    '''def find_governer_races(html):
    xml = requests.get(html).text
    soup = BeautifulSoup(xml,"html.parser")
    result = [a.attrs.get("href","") for a in soup.findAll("a")]
    result = [l for l in result if is_gov_race(l)]
    result = list(set(result))
    return result
print find_governer_races("https://www.realclearpolitics.com/epolls/2010/governor/2010_elections_governor_map.html")
'''
    xml = requests.get(html).text
    dom = web.Element(xml)
    links = [a.attributes.get('href',"") for a in dom.by_tag('a')]
    links = ["https://www.realclearpolitics.com"+str(l) for l in links if is_gov_race(l)]
    #eliminate duplicates!
    links = list(set(links))
    return links
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
    '''result = {}
    dom = web.Element(xml)
    dates = dom.by_tag("series")[0]
    dates = {n.attributes["xid"]:str(n.content) for n in dates.by_tag("value")}
    result["date"] = pd.to_datetime([dates[k] for k in dates.keys()])
    for value in dom.by_tag("graph"):
        name = value.attributes['title']
        dic = {n.attributes["xid"]:float(n.content) if n.content else np.nan for n in value.by_tag("value")}
        result[name] = [dic[k] for k in dic.keys()]
    result = pd.DataFrame(result)
    result = result.sort_values("date",ascending=True)
    result = result.reset_index(drop=True)
    return result'''
    dom = web.Element(xml)
    result = {}
    general = dom.by_tag("div#polling-data-full")[0]
    if_name = general.by_tag("th")
    if len(if_name)>6:
        dates = [date.content.split("-")[1].strip()+"/10" for date in general.by_tag("td")[15::7]]
        dates = pd.to_datetime(dates)
        name_1 = str(if_name[3].content.split("(")[0].strip())
        name_2 = str(if_name[4].content.split("(")[0].strip())
        name_3 = str(if_name[5].content.split("(")[0].strip())
        result["date"] = dates
        result[name_1] = [float(c.content) if c.content!="--" else np.nan for c in general.by_tag("td")[17::7]]
        result[name_2] = [float(c.content) if c.content!="--" else np.nan for c in general.by_tag("td")[18::7]]
        result[name_3] = [float(c.content) if c.content!="--" else np.nan for c in general.by_tag("td")[19::7]]
    elif len(if_name)<=6:
        dates = [date.content.split("-")[1].strip()+"/10" for date in general.by_tag("td")[13::6]]
        dates = pd.to_datetime(dates)
        name_1 = str(if_name[3].content.split("(")[0].strip())
        name_2 = str(if_name[4].content.split("(")[0].strip())
        result["date"] = dates
        result[name_1] = [float(c.content) for c in general.by_tag("td")[15::6]]
        result[name_2] = [float(c.content) for c in general.by_tag("td")[16::6]]
    result = pd.DataFrame(result)
    return result
def poll_plot(url):
    xml = requests.get(url).text
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
def race_results(url):
    dom = web.Element(requests.get(url).text)
    if dom.by_tag('div#polling-data-rcp') == True:
        table = dom.by_tag('div#polling-data-rcp')[0]
        result_data = table.by_tag("tr.final")[0]
        result_data = [float(n.content) for n in result_data.by_tag("td")[3:-1]]
        tot = sum(result_data) / 100
        result_name = [str(n.content).split("(")[0].strip() for n in table.by_tag("th")[3:-1]]
        result = {name:data/tot for name,data in zip(result_name,result_data)}
    else:
        table = dom.by_tag('div#polling-data-full')[0]
        result_data = table.by_tag("tr.final")[0]
        result_data = [float(n.content) for n in result_data.by_tag("td")[3:-1]]
        tot = sum(result_data) / 100
        result_name = [str(n.content).split("(")[0].strip() for n in table.by_tag("th")[3:-1]]
        result = {name:round(data/tot,2)for name,data in zip(result_name,result_data)}
    return result
#page ='http://www.realclearpolitics.com/epolls/2010/governor/2010_elections_governor_map.html'
#for race in find_governor_races(page):
#    print race_results(race)
def id_from_url(url):
    return str(url).split("-")[-1].split(".html")[0]
def plot_race(url):
    xml = requests.get(url).text
    colors = plot_colors(xml)
    if len(colors) == 0:
        return
    result =race_results(url)
    poll_plot(url)
    plt.xlabel("Date")
    plt.ylabel("Polling Percentage")
    for r in result:
        plt.axhline(result[r], color=colors[_strip(r)], alpha=0.6, ls='--')
#for race in find_governor_races(page):
 #   plot_race(race)
 #   plt.show()
def party_from_color(color):
    if color in ['#0000CC', '#3B5998']:
        return "democrat"
    if color in ['#FF0000', '#D30015']:
        return "republican"
    return "other"
#url = "http://www.realclearpolitics.com/epolls/2010/governor/2010_elections_governor_map.html"
url = "https://www.realclearpolitics.com/epolls/2010/governor/md/maryland_governor_ehrlich_vs_omalley-1121.html"
def error_data(url):
    xml = requests.get(url).text
    colors = plot_colors(xml)
    if len(colors) ==0:
        return pd.DataFrame()
    df = rcp_poll_data(xml)
    result = race_results(url)
    #df = df.rename(columns={c:_strip(c) for c in df.columns})
    #for k,v in result.items():
        #result[_strip(k)] = v
    candidates = [c for c in df.columns if c is not "date"]
    df.index = df.date
    df = df.resample("D").mean()
    df = df.dropna()

    forecast_length = (df.index.max() - df.index).values
    forecast_length = forecast_length / np.timedelta64(1,"D")
    errors = {}
    normalized = {}
    poll_lead = {}
    for c in candidates:
        corr = df[c].values / df[candidates].sum(axis=1).values*100
        err = corr - result[c]
        normalized[c] = corr
        errors[c] = err
    n = forecast_length.size
    result = {}
    result["percentage"] = np.hstack(normalized[c] for c in candidates)
    result["error"] = np.hstack(errors[c] for c in candidates)
    result["candidate"] = np.hstack(np.repeat(c,n) for c in candidates)
    result["party"] = np.hstack(np.repeat(party_from_color(colors[c]),n) for c in candidates)
    result["forecast_length"] = np.hstack(forecast_length for _ in candidates)
    result = pd.DataFrame(result)
    return result
#print error_data(url)
'''def all_error_data():
    data = [error_data(race_page) for race_page in find_governor_races(url)]
    return pd.concat(data,ignore_index=True)
errors = all_error_data()
errors.error.hist(bins=50)
plt.xlabel("Polling Error")
plt.ylabel("N")
plt.show()'''
errors = error_data(url)
def bootstrap_result(c1,c2,errors,nsample=10000):
    tot = (c1+c2)
    c1 = 100.*c1/tot
    c2 = 100.*c2/tot

    indices = np.random.randint(0,errors.shape[0], nsample)
    errors = errors.error.iloc[indices].values

    c1_actual = c1 - errors
    c2_actual = c2 + errors

    p1 = (c1_actual>c2_actual).mean()
    p2 = 1-p1
    return p1,p2
nsample = 10000
mcauliffe, cuccinelli = 42.3, 55.8
pm,pc =  bootstrap_result(mcauliffe,cuccinelli,errors,nsample=nsample)
'''print "Georgia Race"
print "-----------------"
print "P(roy wins = %0.2f"%pm
print "P(Nathan wins = %0.2f"%pc'''
print pm,pc
