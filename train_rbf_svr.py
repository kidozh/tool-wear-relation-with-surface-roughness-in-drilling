print(__doc__)

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from data import Data
from sklearn.decomposition import PCA


# Dataset acquisition
data = Data()
X = data.getInputValues()
y = data.getSurfaceRoughnessValues()

SAMPLE_LENGTH = X.shape[0]

# Fit regression model
svr_rbf = SVR(kernel='rbf', C=100)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)

# #############################################################################
# Look at the results
lw = 2

svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ['RBF', 'Linear', 'Polynomial']
model_color = ['m', 'c', 'g']

# 3-fold train data
kf = RepeatedKFold(n_splits=5)
x_np_array, y_np_array = X.to_numpy(), y.to_numpy()
index = np.arange(0, SAMPLE_LENGTH)
for train_index, test_index in kf.split(X):
    train_x, train_y = x_np_array[train_index], y_np_array[train_index]
    test_x, test_y = x_np_array[test_index], y_np_array[test_index]
    clf = svr_rbf.fit(train_x, train_y)

    mae_in_train = mean_absolute_error(svr_rbf.predict(train_x), train_y)
    mae_in_test = mean_absolute_error(svr_rbf.predict(test_x), test_y)
    r2_score_in_train = r2_score(svr_rbf.predict(train_x), train_y)
    # r2_score_in_test = r2_score(svr_rbf.predict(test_x),test_y)
    print(mae_in_train,mae_in_test)
    plt.plot(index, y, label="Real surface roughness")
    plt.scatter(train_index, svr_rbf.predict(train_x),facecolor="none",edgecolor="k", label="SF in train dataset")
    plt.scatter(test_index, svr_rbf.predict(test_x),facecolor="orange", label="SF in test dataset")
    plt.xlabel("Holes index")
    plt.ylabel("Surface roughness[$\mu m$]")
    plt.legend()
    plt.show()

    # visualize using matplotlib
    z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]
    tmp = np.linspace(-5, 5, 30)
    x, y = np.meshgrid(tmp, tmp)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot3D(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2], 'ob')
    # ax.plot3D(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], 'sr')
    ax.plot_surface(x, y, z(x, y))
    ax.view_init(30, 60)
    plt.show()

# svr_rbf.fit(X, y)

