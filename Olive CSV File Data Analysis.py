import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import requests
#from pattern import web
from collections import defaultdict
#import brewer2mpl
import pandas as pd
#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
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



'''
mean = data_groupby.aggregate(lambda x:x.mean())
standard = data_groupby.aggregate(lambda x:x.std())
standard.rename(columns={c:c+"_std" for c in standard.columns},inplace=True)
mean.rename(columns={c:c+"_mean" for c in mean.columns},inplace=True)
mean_palmitic = mean[["palmitic_mean"]]
standard_palmitic = standard[["palmitic_std"]]
new = standard_palmitic.join(mean_palmitic)
print new
weights = np.random.uniform(size=data.shape[0])
other = data[["region"]]
theother = data[["palmitic"]]
other = other.join(theother)
other["weight"] = weights
#print other.head()

def wfunc(f):
    return (f.palmitic*f.weight).sum()/f.weight.sum()
#print other.groupby("region").apply(wfunc)
'''
def col_sum(column):
    return np.sum(column)
#print other.aggregate(col_sum)

keys = [1,2,3]
val  = ["South","Sardinia","North"]
sum = {a:b for a,b in zip(keys,val)}
#print sum
data = pd.read_csv("C:\\Users\\lenovo\\Desktop\\KODLAR\\data\\olive.csv")
data.rename(columns={data.columns[0]:"Areastring"},inplace=True)
data.Areastring = data.Areastring.apply(lambda x:x.split(".")[1])
groupby_region = data.groupby("region")
#print groupby_region.head()
mean_aggregate = groupby_region.aggregate(np.mean)
acidlist = ['palmitic', 'palmitoleic', 'stearic', 'oleic', 'linoleic', 'linolenic', 'arachidic', 'eicosenoic']
#mean_aggregate = mean_aggregate[acidlist]
#print mean_aggregate
'''ax = mean_aggregate.plot(kind="barh",stacked=True)
ax.set_title("Lavuk")
ax.set_yticklabels(val)
ax.set_xlim([0,10000])
plt.tight_layout()
plt.show()
'''
'''mask = (data.area>5)
#print mask
print np.sum(mask)
print np.mean(mask)
lowarea = data[data.area>5]
#print lowarea
print pd.crosstab(lowarea.area,lowarea.region)'''
#Homework-2 Deneme----------------------------------------------------------------------------------------------------------
predictwise = pd.read_csv("C:\\Users\\lenovo\\Desktop\\KODLAR\\data\\predictwise.csv").set_index("States")
electoral_votes = pd.read_csv("C:\\Users\\lenovo\\Desktop\\KODLAR\\data\\electoral_votes.csv").set_index("State")
#print predictwise
def simulate_election(model,n_sim):
    simulation = np.random.uniform(size=(51,n_sim))
    obama_votes = (simulation<model.Obama.values.reshape(-1,1))*model.Votes.values.reshape(-1,1)
    return obama_votes.sum(axis=0)
#print simulate_election(predictwise,1000)
def plot_simulation(simulation):
    plt.hist(simulation,bins=np.arange(200,538,1),label="simulation",color="g",normed=True,align="left",edgecolor="k",linewidth=0.4)
    plt.axvline(269,0,.5,color="k",label="Victory Threshold")
    plt.axvline(332,0,.5,color="r",label="Actual outcome")
    p05 = np.percentile(simulation,5.)
    p95 = np.percentile(simulation,95.)
    iq = int(p95-p05)
    wiq = (simulation>=269).mean()*100
    plt.title("Win Percentage %0.2f and %0.2f Votes"%(wiq,iq))
    plt.legend()
    plt.xticks(np.arange(200,538,25))
    plt.xlabel("Obama Electoral College Votes")
    plt.ylabel("Probability")
#plot_simulation(simulate_election(predictwise,1000))
#plt.show()

#QUESTION 3_________________________________________________________________________________________________________________________________________
'''data = pd.DataFrame(dict(resp=np.arange(1,6),A=["poor","good","very good","bad","very bad"],B=["poor","poor","very good","poor","very bad"],C=["very poor","good","good","bad","very good"]))
#col1 = list(data.columns.values)
#col1 = ["resp","A","B","C"]
data.set_index("resp",inplace=True)
print(data)
print(data.replace("good"))
print(data["A"])'''
#Sum training
data = pd.DataFrame(dict(A=[0,1,2,3],B=[3,4,2,2],C=[5,3,4,2]))
#print(data)
#print(data.sum())

#data içne yazılma training
a = data.A
b = data.B
data[a==0] = 2*b[a==0]

#reindex ve sort
'''df = pd.DataFrame({'A':[1,2,3],
                   'B':[4,5,6],
                   'colname':['7','3','9'],
                   'D':[1,3,5],
                   'E':[5,3,6],
                   'F':[7,4,3]})
print(df.colname.astype(int))
print(df.colname.astype(int).sort_values())
print(df.reindex(df.colname.astype(int).sort_values().index))
print(df.reindex(df.colname.astype(int).sort_values().index).reset_index(drop=True))'''

#matplotlib-hist Training
s = np.random.uniform(-1,0,1000)
#plt.hist(s,15,normed=True,edgecolor="k")
#plt.plot(s,np.ones_like(s),color="r",linewidth=2.)
#plt.show()

#polyfit-polyval-----------------------------------------------------------------------------------------------------------------
#print(np.random.normal(1,2))
def generate_curve(x, sigma):
    return np.random.normal(10 - 1. / (x + 0.1), sigma)

x = np.linspace(0,5,5)
#plt.scatter(x,x**2)
fit = np.polyfit(x,x**2,1)
#plt.plot(x,np.polyval(fit,x))
#plt.show()
#train_test_split__---------------------------------------------------------------------------------------
from sklearn.cross_validation import train_test_split
a,b = np.arange(10).reshape((5,2)),range(5)
x_train,x_test,y_train,y_test = train_test_split(a,b,test_size=0.2,random_state=0)
#print(x_test)
#print(x_train)
#print(y_train)
#print(y_test)
#training
N = 200
#plt.plot([0,N],[1,1],"k--",lw=2)
#plt.show()
#__________________________________________---------------------------------------------------------------
array = np.array([[1,2],[3,4],[5,6]])
#print(array.reshape(1,-1).flatten())

#skelarn.linearresgression----
from sklearn.linear_model import LinearRegression
x = np.random.randn(10,1)
y = 3*x+3+0.1*np.random.randn(10,1)
#plt.scatter(x,y,lw=.5,edgecolors="k",color="r",s=20,label="observed data")
model = LinearRegression()
model.fit(x,y)
#print(model.predict(20))
#x_test = np.linspace(-3,3)
y_pred = model.predict(x.reshape(-1,1))
#print(y_pred)
#print(model.intercept_)
#print(model.coef_)

#Regression a bit more complicated func------------------------------------------
''' np.linspace(-5,5,100)[:,None]
y = -0.5 + 2.2*x + 0.3*x**3 + 2*np.random.randn(100,1)

x_new = np.hstack([x,x**2,x**3,x**4])
model = LinearRegression()
print(model.fit(x_new,y))
print(model.coef_)
print(model.intercept_)

y_pred = model.predict(x_new)
plt.scatter(x,y,color="r",edgecolors="k",lw=.5)
plt.plot(x_new[:,0],y_pred)
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4,include_bias=False)
n_new2 = poly.fit_transform(x)
print(n_new2[:2])'''

#Soru Çözümü---------------------------------------------------------------------------------------------------------------------------------

censure_data = pd.read_csv("C:\\Users\\lenovo\\Desktop\\KODLAR\\content-master\\labs\\lab4\\data\\census_demographics.csv")
def capitalize(s):
    s = s.title()
    s = s.replace("Of","of")
    return s
censure_data["State"] = censure_data.state.apply(capitalize)
del censure_data["state"]
censure_data["State"] = censure_data["State"].replace(abbrev_states_dict)
censure_data = censure_data.set_index("State")
#print(censure_data.head())

smaller_frame = censure_data[["educ_coll","average_income","per_vote"]]
from pandas.plotting import scatter_matrix
#axeslist = scatter_matrix(smaller_frame,alpha=.8,figsize=(9,6),color=["b","g","k"],diagonal="kde")
#for ax in axeslist.flatten():
#    ax.grid(False)
#plt.tight_layout()
#plt.show()

#Linear Regression Example of Lab4--------------------------------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
X_HD = smaller_frame[["educ_coll","average_income"]].values
X_HDn = (X_HD - X_HD.mean(axis=0))/X_HD.std(axis=0)
educ_coll_std_vec = X_HDn[:,0]
educ_coll_std = educ_coll_std_vec.reshape(-1,1)
average_income_std_vec = X_HDn[:,1]
average_income_std = average_income_std_vec.reshape(-1,1)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(educ_coll_std,average_income_std,test_size=0.25)

clf1 = LinearRegression()
clf1.fit(X_train,y_train)

y_predicted = clf1.predict(X_train)
y1_predicted = clf1.predict(X_test)

#print(clf1.coef_,clf1.intercept_)

'''plt.scatter(educ_coll_std,average_income_std,s=20,edgecolors="k",color="y")
plt.plot(X_train,y_predicted,lw=1,color="r")
plt.plot(X_test,y1_predicted,lw=1,color="b")
plt.grid(linestyle="--",color="k",lw=.2)
plt.tight_layout()'''
#plt.show()

#Space Shuttle---------------------------------------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
x = np.random.randn(3,4)
#print(x[0])
clf = LogisticRegression(C=1000)

#Scikit-Learn Understanding IMPORTANT/BEGIN HERE----------------------------------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn import neighbors
iris = load_iris()

x,y = iris.data, iris.target

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(x,y)
result = knn.predict([[3,5,4,2],])
#print(iris.target_names[result])
#print(knn.predict_proba([[3,5,4,2],]))

#SVC Model---------------------------------------------------------------------------------------------------------------
'''from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

model = SVC()
model.fit(x,y)
#print(model.predict(x[::10]))

x = np.random.random(20)
y = 3 * x + 2 + np.random.randn(20)
plt.plot(x,y,"o")'''
#Linear Regression-------------------------------------------------------------------------------------------------------
'''model = LinearRegression()
model.fit(x.reshape(-1,1),y)

X_fit = np.linspace(0,1,100).reshape(-1,1)
Y_fit = model.predict(X_fit)
plt.plot(X_fit.squeeze(),Y_fit.squeeze(),"--")
'''
#RandomForest Regression-------------------------------------------------------------------------------------------------
'''from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(x.reshape(-1,1),y)


X_fit = np.linspace(0,1,100).reshape(-1,1)
Y_fit = model.predict(X_fit)

plt.plot(X_fit,Y_fit,color="b")
#plt.show()
'''
#Loading and vizualization of the digit data----------------------------------------------------------------------------
from sklearn import datasets
digits = datasets.load_digits()
#Plot of the digits ----------------------------------------------------------------------------------------------------
'''fig,axes = plt.subplots(10,10,figsize=(8,8))
fig.subplots_adjust(hspace=.1,wspace=.1)
for i,ax in enumerate(axes.flat):
    ax.imshow(digits.images[i],cmap="binary",interpolation="nearest")
    ax.text(0.05,.05,str(digits.target[i]),
            transform=ax.transAxes,color="green")
    ax.set_xticks([])
    ax.set_yticks([])
#plt.show()'''

'''from sklearn.manifold import Isomap

iso = Isomap(n_components=2)
data_projected = iso.fit_transform(digits.data)

print(data_projected.shape)
print(data_projected[:,0])
plt.scatter(data_projected[:,0],data_projected[:,1],c=digits.target,s=10,
            edgecolor="none",alpha=.5,cmap=plt.cm.get_cmap("nipy_spectral",10))
plt.colorbar(label="dgit label",ticks=range(10))
plt.clim(-0.5,9.5)
#plt.show()

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(digits.data,digits.target,random_state=2)

print(X_train.shape,X_test.shape)

clf = LogisticRegression(penalty="l2")
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,Y_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,Y_pred))

plt.imshow(np.log(confusion_matrix(Y_test,Y_pred)),
           cmap="Blues",interpolation="nearest")
plt.grid(False)
plt.xlabel("Prediction")
plt.ylabel("true")
#plt.show()
'''
#Support Vector Machines------------------------------------------------------------------------------------------------
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50,centers=2,
                  random_state=0,cluster_std=.6)
#plt.scatter(X[:,0],X[:,1],c=y , s=40,cmap="spring",edgecolors="k",linewidth=.5)
#plt.show()

from sklearn.svm import SVC
clf = SVC(kernel="linear")
clf.fit(X,y)

#-----------------------------------------------------------------------------------------------------------------------

import warnings


'''def plot_venn_diagram():
    fig, ax = plt.subplots(subplot_kw=dict(frameon=False, xticks=[], yticks=[]))
    ax.add_patch(plt.Circle((0.3, 0.3), 0.3, fc='red', alpha=0.5))
    ax.add_patch(plt.Circle((0.6, 0.3), 0.3, fc='blue', alpha=0.5))
    ax.add_patch(plt.Rectangle((-0.1, -0.1), 1.1, 0.8, fc='none', ec='black'))
    ax.text(0.2, 0.3, '$x$', size=30, ha='center', va='center')
    ax.text(0.7, 0.3, '$y$', size=30, ha='center', va='center')
    ax.text(0.0, 0.6, '$I$', size=30)
    ax.axis('equal')'''


def plot_example_decision_tree():
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_axes([0, 0, 0.8, 1], frameon=False, xticks=[], yticks=[])
    ax.set_title('Example Decision Tree: Animal Classification', size=24)

    def text(ax, x, y, t, size=20, **kwargs):
        ax.text(x, y, t,
                ha='center', va='center', size=size,
                bbox=dict(boxstyle='round', ec='k', fc='w'), **kwargs)

    text(ax, 0.5, 0.9, "How big is\nthe animal?", 20)
    text(ax, 0.3, 0.6, "Does the animal\nhave horns?", 18)
    text(ax, 0.7, 0.6, "Does the animal\nhave two legs?", 18)
    text(ax, 0.12, 0.3, "Are the horns\nlonger than 10cm?", 14)
    text(ax, 0.38, 0.3, "Is the animal\nwearing a collar?", 14)
    text(ax, 0.62, 0.3, "Does the animal\nhave wings?", 14)
    text(ax, 0.88, 0.3, "Does the animal\nhave a tail?", 14)

    text(ax, 0.4, 0.75, "> 1m", 12, alpha=0.4)
    text(ax, 0.6, 0.75, "< 1m", 12, alpha=0.4)

    text(ax, 0.21, 0.45, "yes", 12, alpha=0.4)
    text(ax, 0.34, 0.45, "no", 12, alpha=0.4)

    text(ax, 0.66, 0.45, "yes", 12, alpha=0.4)
    text(ax, 0.79, 0.45, "no", 12, alpha=0.4)

    ax.plot([0.3, 0.5, 0.7], [0.6, 0.9, 0.6], '-k')
    ax.plot([0.12, 0.3, 0.38], [0.3, 0.6, 0.3], '-k')
    ax.plot([0.62, 0.7, 0.88], [0.3, 0.6, 0.3], '-k')
    ax.plot([0.0, 0.12, 0.20], [0.0, 0.3, 0.0], '--k')
    ax.plot([0.28, 0.38, 0.48], [0.0, 0.3, 0.0], '--k')
    ax.plot([0.52, 0.62, 0.72], [0.0, 0.3, 0.0], '--k')
    ax.plot([0.8, 0.88, 1.0], [0.0, 0.3, 0.0], '--k')
    ax.axis([0, 1, 0, 1])


def visualize_tree(estimator, X, y, boundaries=True,
                   xlim=None, ylim=None):
    estimator.fit(X, y)

    if xlim is None:
        xlim = (X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    if ylim is None:
        ylim = (X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)

    x_min, x_max = xlim
    y_min, y_max = ylim
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, alpha=0.2, cmap='rainbow')
    plt.clim(y.min(), y.max())

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='rainbow',edgecolors="k",lw=.5)
    plt.axis('off')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.clim(y.min(), y.max())

    # Plot the decision boundaries
    def plot_boundaries(i, xlim, ylim):
        if i < 0:
            return

        tree = estimator.tree_

        if tree.feature[i] == 0:
            plt.plot([tree.threshold[i], tree.threshold[i]], ylim, '-k')
            plot_boundaries(tree.children_left[i],
                            [xlim[0], tree.threshold[i]], ylim)
            plot_boundaries(tree.children_right[i],
                            [tree.threshold[i], xlim[1]], ylim)

        elif tree.feature[i] == 1:
            plt.plot(xlim, [tree.threshold[i], tree.threshold[i]], '-k')
            plot_boundaries(tree.children_left[i], xlim,
                            [ylim[0], tree.threshold[i]])
            plot_boundaries(tree.children_right[i], xlim,
                            [tree.threshold[i], ylim[1]])

    if boundaries:
        plot_boundaries(0, plt.xlim(), plt.ylim())


def plot_tree_interactive(X, y):
    from sklearn.tree import DecisionTreeClassifier

    def interactive_tree(depth=1):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        visualize_tree(clf, X, y)

    from IPython.html.widgets import interact
    return interact(interactive_tree, depth=[1, 5])


def plot_kmeans_interactive(min_clusters=1, max_clusters=6):
    from IPython.html.widgets import interact
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.datasets.samples_generator import make_blobs

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        X, y = make_blobs(n_samples=300, centers=4,
                          random_state=0, cluster_std=0.60)

def _kmeans_step(frame=0, n_clusters=4):
            rng = np.random.RandomState(2)
            labels = np.zeros(X.shape[0])
            centers = rng.randn(n_clusters, 2)

            nsteps = frame // 3

            for i in range(nsteps + 1):
                old_centers = centers
                if i < nsteps or frame % 3 > 0:
                    dist = euclidean_distances(X, centers)
                    labels = dist.argmin(1)

                if i < nsteps or frame % 3 > 1:
                    centers = np.array([X[labels == j].mean(0)
                                        for j in range(n_clusters)])
                    nans = np.isnan(centers)
                    centers[nans] = old_centers[nans]


            # plot the data and cluster centers
            plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='rainbow',
                        vmin=0, vmax=n_clusters - 1);
            plt.scatter(old_centers[:, 0], old_centers[:, 1], marker='o',
                        c=np.arange(n_clusters),
                        s=200, cmap='rainbow')
            plt.scatter(old_centers[:, 0], old_centers[:, 1], marker='o',
                        c='black', s=50)

            # plot new centers if third frame
            if frame % 3 == 2:
                for i in range(n_clusters):
                    plt.annotate('', centers[i], old_centers[i],
                                 arrowprops=dict(arrowstyle='->', linewidth=1))
                plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                            c=np.arange(n_clusters),
                            s=200, cmap='rainbow')
                plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                            c='black', s=50)

            plt.xlim(-4, 4)
            plt.ylim(-2, 10)

            if frame % 3 == 1:
                plt.text(3.8, 9.5, "1. Reassign points to nearest centroid",
                         ha='right', va='top', size=14)
            elif frame % 3 == 2:
                plt.text(3.8, 9.5, "2. Update centroids to cluster means",
                         ha='right', va='top', size=14)


    return interact(_kmeans_step, frame=[0, 50],
                    n_clusters=[min_clusters, max_clusters])


def plot_image_components(x, coefficients=None, mean=0, components=None,
                          imshape=(8, 8), n_components=6, fontsize=12):
    if coefficients is None:
        coefficients = x

    if components is None:
        components = np.eye(len(coefficients), len(x))

    mean = np.zeros_like(x) + mean


    fig = plt.figure(figsize=(1.2 * (5 + n_components), 1.2 * 2))
    g = plt.GridSpec(2, 5 + n_components, hspace=0.3)

    def show(i, j, x, title=None):
        ax = fig.add_subplot(g[i, j], xticks=[], yticks=[])
        ax.imshow(x.reshape(imshape), interpolation='nearest')
        if title:
            ax.set_title(title, fontsize=fontsize)

    show(slice(2), slice(2), x, "True")

    approx = mean.copy()
    show(0, 2, np.zeros_like(x) + mean, r'$\mu$')
    show(1, 2, approx, r'$1 \cdot \mu$')

    for i in range(0, n_components):
        approx = approx + coefficients[i] * components[i]
        show(0, i + 3, components[i], r'$c_{0}$'.format(i + 1))
        show(1, i + 3, approx,
             r"${0:.2f} \cdot c_{1}$".format(coefficients[i], i + 1))
        plt.gca().text(0, 1.05, '$+$', ha='right', va='bottom',
                       transform=plt.gca().transAxes, fontsize=fontsize)

    show(slice(2), slice(-2, None), approx, "Approx")


def plot_pca_interactive(data, n_components=6):
    from sklearn.decomposition import PCA
    from IPython.html.widgets import interact

    pca = PCA(n_components=n_components)
    Xproj = pca.fit_transform(data)

    def show_decomp(i=0):
        plot_image_components(data[i], Xproj[i],
                              pca.mean_, pca.components_)

    interact(show_decomp, i=(0, data.shape[0] - 1));

#Random Forest Trees----------------------------------------------------------------------------------------------------
#Decision Trees and over-fitting----------------------------------------------------------------------------------------
X,y = make_blobs(n_samples=300,centers=4,random_state=0,
                 cluster_std=1.)
#plt.scatter(X[:,0],X[:,1],c=y,s=20,cmap="rainbow",edgecolors="k",linewidth=.5)
#plt.show()

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
#visualize_tree(clf,X[:200],y[:200],boundaries=False)

#visualize_tree(clf,X[:-200],y[:-200],boundaries=False)

#Random Forest Classfier----------------------------------------------------------------------------------------------
#Genel bir ortalama alıyor decision tree lerin.

'''from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100,random_state=0)
visualize_tree(clf,X[:200],y[:200],boundaries=False)
plt.show()'''

#Random Tree Regresion--------------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestRegressor
x = 10*np.random.rand(100)

def model(x, sigma=.3):
    fast_oscillation = np.sin(5*x)
    low_oscillation = np.sin(.5*x)
    noise = sigma * np.random.randn(len(x))
    return fast_oscillation + low_oscillation + noise

'''y = model(x)
fig, ax = plt.subplots(figsize=(8,4))
ax = plt.gca()
ax.errorbar(x,y,.3,fmt="o")
ax.grid("w",lw=.2)
ax.set_facecolor("grey")

xfit = np.linspace(0,10,1000)
yfit = RandomForestRegressor(100).fit(x.reshape(-1,1),y).predict(xfit.reshape(-1,1))
ytrue = model(xfit,0)

plt.plot(xfit,yfit,"g")
plt.plot(xfit,ytrue,"m")
#plt.show()'''

#Random Forest for classifying Digits-----------------------------------------------------------------------------------

from sklearn.datasets import load_digits

digits = load_digits()

x = digits.data
y=digits.target
#print(x.shape)
#print(y.shape)

fig = plt.figure(figsize=(10,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=.05,wspace=.05)

'''for i in range(70):
    ax = fig.add_subplot(10,7,i+1,xticks=[],yticks=[])
    ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation="nearest")
    ax.text(0,6,str(digits.target[i]))
plt.tight_layout()
#plt.show()'''

from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=0)
#print(X_train.shape)
clf = DecisionTreeClassifier(max_depth=11)
clf.fit(X_train ,y_train)
ypred = clf.predict(X_test)

a = metrics.accuracy_score(ypred,y_test)
print(a)

plt.imshow(metrics.confusion_matrix(ypred,y_test),interpolation="nearest",cmap=plt.cm.binary)
plt.grid(False)
plt.colorbar()
plt.xlabel("predicted label")
plt.ylabel("True label")
plt.tight_layout()
#plt.show()

clf1 = RandomForestClassifier(n_estimators=100,max_depth=16)
clf1.fit(X_train,y_train)
ypred1 = clf1.predict(X_test)

b = metrics.accuracy_score(ypred1,y_test)
print(b)

clf2 = SVC(C=1.,kernel="rbf",gamma="auto")
clf.fit(X_train,y_train)

#2nd VIDEO OF JAKEVANDER STARTS------------------------------------------------------------------------------------------



