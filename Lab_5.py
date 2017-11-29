#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import brewer2mpl
from matplotlib import rcParams

#colorbrewer2 Dark2 qualitative color table
dark2_cmap = brewer2mpl.get_map('Dark2', 'Qualitative', 7)
dark2_colors = dark2_cmap.mpl_colors

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
import warnings
warnings.filterwarnings('ignore', message='Polyfit*')
#-----------------------------------------------------------------------------------------------------------------
import random
import copy
def scatter_by(df, scatterx, scattery, by=None, figure=None, axes=None, colorscale=dark2_cmap, labeler={}, mfunc=None, setupfunc=None, mms=8):
    cs=copy.deepcopy(colorscale.mpl_colors)
    if not figure:
        figure=plt.figure(figsize=(8,8))
    if not axes:
        axes=figure.gca()
    x=df[scatterx]
    y=df[scattery]
    if not by:
        col=random.choice(cs)
        axes.scatter(x, y, cmap=colorscale, c=col)
        if setupfunc:
            axeslist=setupfunc(axes, figure)
        else:
            axeslist=[axes]
        if mfunc:
            mfunc(axeslist,x,y,color=col, mms=mms)
    else:
        cs=list(np.linspace(0,1,len(df.groupby(by))))
        xlimsd={}
        ylimsd={}
        xs={}
        ys={}
        cold={}
        for k,g in df.groupby(by):
            col=cs.pop()
            x=g[scatterx]
            y=g[scattery]
            xs[k]=x
            ys[k]=y
            c=colorscale.mpl_colormap(col)
            cold[k]=c
            axes.scatter(x, y, c=c, label=labeler.get(k,k), s=40, alpha=0.3);
            xlimsd[k]=axes.get_xlim()
            ylimsd[k]=axes.get_ylim()
        xlims=[min([xlimsd[k][0] for k in xlimsd.keys()]), max([xlimsd[k][1] for k in xlimsd.keys()])]
        ylims=[min([ylimsd[k][0] for k in ylimsd.keys()]), max([ylimsd[k][1] for k in ylimsd.keys()])]
        axes.set_xlim(xlims)
        axes.set_ylim(ylims)
        if setupfunc:
            axeslist=setupfunc(axes, figure)
        else:
            axeslist=[axes]
        if mfunc:
            for k in xs.keys():
                mfunc(axeslist,xs[k],ys[k],color=cold[k], mms=mms);
    axes.set_xlabel(scatterx);
    axes.set_ylabel(scattery);

    return axes

def make_rug(axeslist, x, y, color='b', mms=8):
    axes=axeslist[0]
    zerosx1=np.zeros(len(x))
    zerosx2=np.zeros(len(x))
    xlims=axes.get_xlim()
    ylims=axes.get_ylim()
    zerosx1.fill(ylims[1])
    zerosx2.fill(xlims[1])
    axes.plot(x, zerosx1, marker='|', color=color, ms=mms)
    axes.plot(zerosx2, y, marker='_', color=color, ms=mms)
    axes.set_xlim(xlims)
    axes.set_ylim(ylims)
    return axes
#polynomial Regression---------------------------------------------------------------------------------------
np.random.seed(42)
def rmse(p,x,y):
    yfit = np.polyval(p,x)
    return np.sqrt(np.mean((y-yfit)**2))
def generate_curve(x,sigma):
    return np.random.normal(10-1./(x+.1),sigma)
x = 10**np.linspace(-1,0,8)
intrinsic_error = 1.
y = generate_curve(x,intrinsic_error)
#plt.scatter(x,y,s=20)
#plt.show()

'''x_new = np.linspace(-.2,1.2,1000)
plt.scatter(x,y,s=50)
f1 = np.polyfit(x,y,1)
plt.plot(x_new,np.polyval(f1,x_new))
print("d=1,rmse=",rmse(f1,x,y))
f2=np.polyfit(x,y,2)
plt.plot(x_new,np.polyval(f2,x_new))
print ("d=2, rmse=",rmse(f2,x,y))
f4=np.polyfit(x,y,4)
plt.plot(x_new,np.polyval(f4,x_new))
print ("d=4, rmse=",rmse(f4,x,y))
f6=np.polyfit(x,y,6)
plt.plot(x_new,np.polyval(f6,x_new))
print ("d=6, rmse=",rmse(f6,x,y))
#plt.xlim(-0.2, 1.2)
#plt.ylim(-1, 12)
#plt.show()'''

#Costructing a dataset_______________________________________________________________________________________-
N = 200
x = np.random.random(N)
y = generate_curve(x,intrinsic_error)
#plt.scatter(x,y,s=5,lw=.4,color="m",edgecolors="k")
#plt.grid(linestyle="--",linewidth=.2,color="k")
#
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)
#plt.scatter(x_train,y_train,s=10,lw=.4,color="r",edgecolors="k")
#plt.scatter(x_test,y_test,s=10,lw=.4,color="m",edgecolors="k")
fit = np.polyfit(x_train,y_train,1)
fit1 = np.polyfit(x_test,y_test,1)
#plt.plot(x_train,np.polyval(fit,x_train))
#plt.plot(x_test,np.polyval(fit1,x_test))
#plt.show()

#

ds = np.arange(21)
train_err = np.zeros(len(ds))
test_err = np.zeros(len(ds))

for i,d in enumerate(ds):
    fit = np.polyfit(x_train,y_train,d)
    train_err[i] = rmse(fit,x_train,y_train)
    test_err[i] = rmse(fit,x_test,y_test)

#

'''fig, ax = plt.subplots(figsize=(6,4))
ax.plot(ds,train_err,lw=1,label="train error")
ax.plot(ds,test_err,lw=1,label="test error")
ax.legend(loc=0)
ax.set_xlabel("degree of fit")
ax.set_ylabel("rms error")
plt.tight_layout()
plt.grid(linestyle="--",linewidth=.2,color="k")
plt.xticks(np.arange(0,21,1))'''

#
N = 200
x = np.random.random(N)
y = generate_curve(x,intrinsic_error)

def plot_learning_curve(d):
    sizes = np.linspace(2,N,50).astype(int)
    train_err = np.zeros(sizes.shape)
    crossval_err = np.zeros(sizes.shape)

    for i,size in enumerate(sizes):
        p = np.polyfit(x_train[:size],y_train[:size],d)
        crossval_err[i] = rmse(p,x_test[:size],y_test[:size])
        train_err[i] = rmse(p,x_train,y_train)


    fig,ax = plt.subplots()
    ax.plot(sizes,crossval_err,lw=2,label="validation error")
    ax.plot(sizes,train_err,lw=2,label="train error")
    ax.plot([0,N],[intrinsic_error,intrinsic_error],"k--",label="intrinsic error")

    ax.set_xlabel("training set size")
    ax.set_ylabel("rms error")
    ax.legend(loc=0)

'''plot_learning_curve(5)
plot_learning_curve(10)
plt.ylim((0,10))'''
#plt.show()

#K-Nearest Neighbors-----------------------------------------------------------------------------------------------------
df =pd.read_csv("C:\\Users\\lenovo\\Desktop\\KODLAR\\data\\olive.csv")
df.rename(columns={df.columns[0]:"areastring"},inplace=True)
df["areastring"] = df["areastring"].map(lambda x:x.split(".")[1])
acidlist = ["palmitic","palmitoleic" , "stearic" ,"oleic", "linoleic","linolenic","arachidic", "eicosenoic"]
dfsub = df[acidlist].apply(lambda x:x/100.)
df[acidlist] = dfsub
dfsouth = df[df["region"]==2]
dfsouthns = dfsouth[df.area!=4]

akeys = [1,2,3]
aval = ["North-Apulia","Calabria","South-Apulia"]
amap = {a:b for a,b in zip(akeys,aval)}
ax = scatter_by(dfsouthns,"palmitic","palmitoleic",by="area",mfunc=make_rug,mms=10)
ax.grid(False)
ax.legend(loc=0)


from matplotlib.colors import ListedColormap
#cm_bright = ListedColormap(['#FF0000', '#000000','#0000FF'])
#cm = plt.cm.RdBu
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def points_plot(X, Xtr, Xte, ytr, yte, clf, colorscale=cmap_light, cdiscrete=cmap_bold):
    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))

    plt.figure()
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=0.2)
    plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr-1, cmap=cdiscrete, s=50, alpha=0.2,edgecolor="k")
    # and testing points
    yact=clf.predict(Xte)
    print ("SCORE", clf.score(Xte, yte))
    plt.scatter(Xte[:, 0], Xte[:, 1], c=yte-1, cmap=cdiscrete, alpha=0.5, marker="s", s=35)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    return ax

from sklearn.neighbors import KNeighborsClassifier
subdf=dfsouthns[['palmitic','palmitoleic']]
subdfstd=(subdf - subdf.mean())/subdf.std()
X=subdfstd.values
y=dfsouthns['area'].values
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.6)
Xtr=np.concatenate((Xtrain, Xtest))

clf = KNeighborsClassifier(20).fit(Xtrain, ytrain)
points_plot(Xtr, Xtrain, Xtest, ytrain, ytest, clf)
plt.show()

