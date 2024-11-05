from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import time

def get_datas():
    pointcount = input("Amount of points to calculate type all for all points \n")

    if pointcount == "all":
        pointcount = 1797

    digits = load_digits()
    datas = digits.data[:int(pointcount)]
    target = digits.target[:int(pointcount)]

    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(datas)
    dataformat = data_2d.tolist()
    print("Data converted to 2d. Time: "+str(time.process_time()))
    return pointcount, datas, target, data_2d, dataformat