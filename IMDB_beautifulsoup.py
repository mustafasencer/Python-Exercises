from bs4 import BeautifulSoup
from pattern import web
import requests
import re
import pandas as pd
def _strip(s):
    return re.sub(r'\n',"",s)

url =  "http://www.imdb.com/search/title?sort=num_votes,desc&start=1&title_type=feature&year=1950,2012"
r = requests.get(url).text
soup = BeautifulSoup(r,"html.parser")

title = [];genres = [];runtime = [];rating = []
result = {}
for link in soup.findAll("div",{"class":"lister-item-content"}):
    title.append(link.find("a").contents)
    genres.append(_strip(link.find("span",{"class":"genre"}).get_text()))
    runtime.append(link.find("span",{"class":"runtime"}).get_text())
    rating.append(link.find("strong").get_text())

result = {"title":title,"genre":genres,"runtime":runtime,"rating":rating}
result = pd.DataFrame(result)
print result[result.columns[::-1]]




