import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import SGDClassifier
from sklearn.datasets.samples_generator import make_blobs
import seaborn; seaborn.set()
def plot_venn_diagram():
    fig, ax = plt.subplots(subplot_kw=dict(frameon=False, xticks=[], yticks=[]))
    ax.add_patch(plt.Circle((0.3, 0.3), 0.3, fc='red', alpha=0.5))
    ax.add_patch(plt.Circle((0.6, 0.3), 0.3, fc='blue', alpha=0.5))
    ax.add_patch(plt.Rectangle((-0.1, -0.1), 1.1, 0.8, fc='none', ec='black'))
    ax.text(0.2, 0.3, '$x$', size=30, ha='center', va='center')
    ax.text(0.7, 0.3, '$y$', size=30, ha='center', va='center')
    ax.text(0.0, 0.6, '$I$', size=30)
    ax.axis('equal')

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

'''def plot_tree_interactive(X, y):
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
'''
def plot_sgd_separator():
    # we create 50 separable points
    X, Y = make_blobs(n_samples=50, centers=2,
                      random_state=0, cluster_std=0.60)

    # fit the model
    clf = SGDClassifier(loss="hinge", alpha=0.01,
                        n_iter=200, fit_intercept=True)
    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane
    xx = np.linspace(-1, 5, 10)
    yy = np.linspace(-1, 5, 10)

    X1, X2 = np.meshgrid(xx, yy)
    Z = np.empty(X1.shape)
    for (i, j), val in np.ndenumerate(X1):
        x1 = val
        x2 = X2[i, j]
        p = clf.decision_function([x1, x2])
        Z[i, j] = p[0]
    levels = [-1.0, 0.0, 1.0]
    linestyles = ['dashed', 'solid', 'dashed']
    colors = 'k'

    ax = plt.axes()
    ax.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

    ax.axis('tight')


'''if __name__ == '__main__':
    plot_sgd_separator()
    #plt.show()
'''

#START OF THE COURSE---------------------------------------------------------------------------------------------

from sklearn.datasets import load_iris

iris = load_iris()
#print(iris.keys())
#print(iris.data.shape)
#print(iris.target)
#print(iris.target_names)
#Plot the iris data following the features and target_names------------------

x_index = 0
y_index = 1

'''formatter = plt.FuncFormatter(lambda i,*args:iris.target_names[int(i)])
plt.scatter(iris.data[:,x_index],iris.data[:,y_index],c=iris.target,
            cmap=plt.cm.get_cmap("RdYlBu",3),lw=.3,s=20,edgecolors="k")
plt.colorbar(ticks=[0,1,2], format=formatter)
plt.clim(-0.5,2.5)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.tight_layout()'''
#plt.show()

#Linear Regression-------------------------------------------------
from sklearn.linear_model import LinearRegression

model = LinearRegression(normalize=True)
#print(model.normalize)

x = np.arange(10)
y = 2*x+1
#plt.plot(x,y,"o")

model.fit(x.reshape(-1,1),y)
#print(  model.coef_)
#print(model.intercept_)

#SUPERVISED LEARNING Classification and Regression---------------------------
from sklearn import neighbors,datasets

iris = load_iris()
X,y = iris.data, iris.target

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
target_pred = knn.predict([[3,5,4,2]])
#print(target_pred)
#print(iris.target_names[target_pred])
#print(knn.predict_proba([[3,4,5,2]]))

#SVC Example with İRİS DATA---------------------------------------------------------------
from sklearn.svm import SVC

model = SVC()
model.fit(X, y)
print(model.predict([[3,4,5,2]]))

#RandomForestRegressor Example-----------------------------------------------
from sklearn.ensemble import RandomForestRegressor
X = np.random.random((20,1))
y = 3 * X.flatten() + 2 + np.random.randn(20)

model = LinearRegression()
model.fit(X,y)
X_fit = np.linspace(0,1,100).reshape(-1,1)
y_fit = model.predict(X_fit)
#plt.scatter(X,y,s=30,edgecolors="k",lw=.3)
#plt.plot(X_fit,y_fit,c="g")

model1 = RandomForestRegressor(n_estimators=10,max_depth=5)
model1.fit(X,y)

X_fit1 = np.linspace(0,1,100).reshape(-1,1)
y_fit1 = model1.predict(X_fit1)


#plt.plot(X_fit1,y_fit1,c="m")
#plt.show()

#Unsupervized Learning: Dimensionality Reduction and Clustering------------------

#Dimensionalty Reduction----------------

X, y = iris.data, iris.target

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)
X_reduced = pca.transform(X)
#print(X_reduced.shape)

#plt.scatter(X_reduced[:, 0], X_reduced[:,1], c=y,
#            cmap="RdYlBu", edgecolors="k", lw=.3, s=20)

#print("Meaning of the 2 components")
'''for component in pca.components_:
    print("+".join("%.3f x %s" % (name, value)
                   for name, value in
                   zip(component, iris.feature_names)))
'''
#Clustering Algorithm/Kmeans-----------------------------------------------------

from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3, random_state=0)
k_means.fit(X)
y_pred = k_means.predict(X)

'''plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
            c=y_pred,s=20,edgecolors="k",lw=.3, cmap="RdYlBu")
plt.show()'''

#MODEL VALIDATION------------------------------------------------------------
#KNeighbor Classification
#Not a good example because if we compare the data the model has aldready seen
#it wont make any sense. Train test will go now into action
X, y = iris.data, iris.target
clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X[:100],y[:100])
y_pred = clf.predict(X[100:])
#print(np.all(y[100:] == y_pred))

#Traintestsplit---------------------------
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

X_train,X_test,Y_train,Y_test = train_test_split(X, y)
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
#print(confusion_matrix(Y_test,Y_pred))

#Training on the digits-------------------------------------------------

from sklearn.manifold import Isomap
from sklearn.datasets import load_digits
digits = load_digits()
iso = Isomap(n_components=2)
data_projected = iso.fit_transform(digits.data)
#print(data_projected.shape)

#plot the data transformed from 64 dim to 2 dim.

'''plt.scatter(data_projected[:,0],data_projected[:,1],c=digits.target,
            edgecolors="k", lw=.1, alpha=.5,s=10, cmap=plt.cm.get_cmap("nipy_spectral",10))
plt.colorbar(label="digit label", ticks=range(10))
plt.clim(-.5,9.5)'''
#plt.show()

#Classification of the digits------------------------------

Xtrain,Xtest,Ytrain,Ytest = train_test_split(digits.data, digits.target,
                                             random_state=2)
#print(Xtrain.shape, Xtest.shape)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty="l2")
clf.fit(Xtrain,Ytrain)
Ypred = clf.predict(Xtest)
from sklearn.metrics import accuracy_score
#print(accuracy_score(Ytest,Ypred))

#print(confusion_matrix(Ytest,Ypred))

#PART 2 SUPPORT VECTOR MACHINE------------------------------------------

from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=50, centers=2,  random_state=0,
                  cluster_std=.6)
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap="spring")
#plt.show()

from sklearn.svm import SVC

clf = SVC(kernel="linear")
clf.fit(X, y)

ypred = clf.predict(X)
#print(np.all(ypred==y))

#PART 3 REGRESSION RANDOM FOREST---------------------------------------

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=.6)
#plt.scatter(X[:,0], X[:,1], c=y, cmap="rainbow", s=20, edgecolors="k", lw=.3)
#plt.colorbar(ticks=range(1,5))
#plt.show()

from sklearn.tree import DecisionTreeClassifier

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, random_state=0)
clf = DecisionTreeClassifier(max_depth=10)
clf.fit(Xtrain, Ytrain)
y_pred = clf.predict(Xtest)

a = accuracy_score(y_pred, Ytest)

#PART 4 PCA in depth------------------------------------------------------------------

X = np.dot(np.random.random(size=(2,2)), np.random.normal(size=(2,200))).T
#print(X.shape)
#plt.scatter(X[:, 0],X[:,1],s=20,lw=.2,edgecolors="k")
#plt.axis("equal")
#plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)
#print(pca.explained_variance_)
#print(pca.components_)

plt.plot(X[:,0],X[:,1],"o")
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector *3* np.sqrt(length)
    plt.plot([0, v[0]],[0, v[1]], "-k", lw=3)
plt.axis("equal")

pca = PCA(.95)
X_trans = pca.fit_transform(X)
#print(X.shape)
#print(X_trans.shape)


