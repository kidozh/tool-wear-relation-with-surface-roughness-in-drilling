import pandas as pd
from sklearn.decomposition import *
import matplotlib.pyplot as plt

DataSetPath = "dataset/Ming_luo_data.csv"


class Data:

    def __init__(self):
        self.pd_dataset = pd.read_csv(DataSetPath)

    def getInputFields(self):
        return self.pd_dataset.columns.drop("Holes").drop("Surface roughness")

    def getSurfaceRoughnessValues(self):
        return self.pd_dataset["Surface roughness"]

    def getInputValues(self):
        return self.pd_dataset.get(self.getInputFields())


if __name__ == "__main__":
    data = Data()
    print(data.getInputFields())
    print(data.getInputValues(),data.getSurfaceRoughnessValues().shape)
    sf = data.getSurfaceRoughnessValues()
    # PCA
    pca = PCA(n_components=2)
    pca.fit(data.getInputValues())
    newX = pca.fit_transform(data.getInputValues())
    print(pca.explained_variance_ratio_)
    print(newX)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(sf.to_numpy().shape,newX.shape)

    # set multiple lines
    SAMPLE_APART_INDEX = [0,9,20,28,35,42]
    for i in range(0,len(SAMPLE_APART_INDEX)-1):
        before, after = SAMPLE_APART_INDEX[i], SAMPLE_APART_INDEX[i+1]
        ax.plot3D(newX[before:after, 0], newX[before:after, 1], sf[before:after], label="{} drilling curve".format(i+1))
        ax.set_xlabel("PCA on x axis")
        ax.set_ylabel("PCA on y axis")
        ax.set_zlabel("Surface roughness[$\mu m$]")

    plt.legend()
    plt.show()
