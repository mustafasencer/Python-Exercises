from collections import defaultdict
import json

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib as mpl

#colorbrewer2 Dark2 qualitative color table
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843)]

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks

    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)

    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

states_abbrev_dict = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}
abbrev_states_dict = {v: k for k, v in states_abbrev_dict.items()}

#Q-1__________________________________________________________________________________________________________

census_data = pd.read_csv("C:\\Users\\lenovo\\Desktop\\KODLAR\\content-master\\labs\\lab4\\data\\census_demographics.csv")

def capitalize(s):
    s = s.title()
    s = s.replace("Of","of")
    return s
census_data["State"] = census_data.state.map(capitalize)
del census_data["state"]
census_data["State"] = census_data["State"].replace(abbrev_states_dict)
census_data.set_index("State",inplace=True)
#print(census_data.head())

smaller_frame = census_data[["educ_coll","average_income","per_vote"]]
from pandas.tools.plotting import scatter_matrix
#axeslist = scatter_matrix(smaller_frame,alpha=0.8,figsize=(12,12),diagonal="kde")
#for ax in axeslist.flatten():
 #   ax.grid(False)
#plt.tight_layout()

#print(smaller_frame.corr())

from sklearn.linear_model import LinearRegression
X_HD = smaller_frame[["educ_coll","average_income"]].values
#print(X_HD)
X_HDn = (X_HD - X_HD.mean(axis=0))/X_HD.std(axis=0)
educ_coll_std_vec = X_HDn[:,0]
educ_col_std = educ_coll_std_vec.reshape(-1,1)
average_income_std_vec = X_HDn[:,1]
average_incme_std = average_income_std_vec.reshape(-1,1)

from sklearn.cross_validation import train_test_split

'''X_train,X_test,Y_train,Y_test = train_test_split(educ_col_std,average_incme_std,test_size=0.25)

clf1 = LinearRegression()
clf1.fit(X_train,Y_train)
predicted_train = clf1.predict(X_train)
predicted_test = clf1.predict(X_test)
#trains = X_train.reshape(1,-1).flatten()
#tests = X_test.reshape(1,-1).flatten()
print(clf1.coef_,clf1.intercept_)'''

'''plt.scatter(educ_coll_std_vec,average_income_std_vec,c="r",s=30,edgecolors="k")
plt.plot(X_train,predicted_train,color="b",alpha=.2)
plt.plot(X_test,predicted_test,color="k",alpha=.6)
plt.xlim(-3,5)
plt.grid(lw=.5,linestyle="--",color="k",alpha=.3)
plt.show()'''

'''plt.scatter(predicted_test,predicted_test - Y_test,s=20,color="b")
plt.scatter(predicted_train,predicted_train - Y_train,s=20,color="g")
plt.grid(lw=.5,linestyle="--",color="k",alpha=.3)
plt.plot([.4,2],[0,0])'''
#plt.show()
#print(clf1.score(X_train,Y_train),clf1.score(X_test,Y_test))

#Doing a PCA on the data______________________________________________-

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = pca.fit_transform(X_HDn)
#print(pca.explained_variance_ratio_)

#Space Shuttle-------------------------------------------------------------------------------------------------

from IPython.display import Image as Im
from IPython.display import display

#i = Im(filename="C:\\Users\\lenovo\\Desktop\\KODLAR\\content-master\\labs\\lab4\\data\\shuttle.png")
#display(i)

data = []
#for item in open("C:\\Users\\lenovo\\Desktop\\KODLAR\\content-master\\labs\\lab4\\data\\chall.txt"):
#    item = [float(a) for a in item.strip().split()]
#    data.append(item)
bigdata = open("C:\\Users\\lenovo\\Desktop\\KODLAR\\content-master\\labs\\lab4\\data\\chall.txt")
#print(np.array(data).reshape(-1,2))
data = np.array(data).reshape(-1,2)
temps,pfail = data[:,0],data[:,1]
#plt.scatter(temps,pfail,color="r",s=10)
plt.grid(False)
remove_border()
plt.ylim(-.2,1.2)
plt.tight_layout()
#plt.show()
print(temps)
from sklearn.linear_model import LogisticRegression
reg = 1000.
clf4 = LogisticRegression(C=reg)
print(clf4)
clf4.fit(temps.reshape(-1,1),pfail)
print(clf4)

tempsnew = np.linspace(20.,90.,15)
probs = clf4.predict_proba(tempsnew.reshape(-1,1))[:,1]
predicts = clf4.predict(tempsnew.reshape(-1,1))

axes = plt.gca()
axes.grid(False)
#axes.plot(tempsnew,probs,marker="s",markeredgecolor="k",markeredgewidth=.2,markersize=5,color="b")
#axes.scatter(tempsnew,predicts,marker="s",s=5,color="g")
remove_border(axes)
#plt.show()

cross = pd.crosstab(pfail,clf4.predict(temps.reshape(-1,1)),rownames=["Actual"],colnames=["Predicted"])
#print(cross)

#my turn logistic regression---------------------------------------------------------------------------------

clf4w = LogisticRegression()
clf4w = clf4w.fit(temps.reshape(-1,1),pfail)

probs = clf4w.predict_proba(temps.reshape(-1,1))[:,1]
predicts = clf4w.predict(temps.reshape(-1,1))
print(predicts)
plt.scatter(temps,pfail,marker="s",color="y",s=10)
plt.plot(temps,probs,markersize=5)
plt.scatter(temps,predicts,marker="s",s=5,color="m")
plt.grid(False)
remove_border()
plt.tight_layout()
#plt.show()

crossw = pd.crosstab(pfail,clf4w.predict(temps.reshape(-1,1)),rownames=["Actual"],colnames=["Predicted"])
data = [[float(j) for j in e.strip().split()] for e in bigdata]
print(crossw)

#Logistic Regression with cross_validation________________________________________________________________________---

from sklearn.linear_model import LogisticRegression

def fit_logistic(X_train,y_train,reg=.0001,penalty="l2"):
    clf = LogisticRegression(C=reg,penalty=penalty)
    clf = clf.fit(X_train,y_train)
    return clf
from sklearn.grid_search import GridSearchCV

def cv_optimize(X_train,y_train,paramslist,penalty="l2",n_folds=10):
    clf = LogisticRegression(penalty=penalty)
    parameters= {"C":paramslist}
    gs = GridSearchCV(clf, param_grid=parameters,cv=n_folds)
    gs = gs.fit(X_train,y_train)
    return gs.best_params_,gs.best_score_

def cv_and_fit(X_train,y_train,paramslist,penalty="l2",n_folds=5):
    bp,bs = cv_optimize(X_train,y_train,paramslist,penalty=penalty,n_folds=n_folds)
    print("BP,BS",bp,bs)
    clf = fit_logistic(X_train,y_train,penalty=penalty,reg=bp["C"])
    return clf

clf = cv_and_fit(temps.reshape(-1,1),pfail,np.logspace(-4,3,num=100))
print(clf)
