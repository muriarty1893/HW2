from numpy import loadtxt, ones, zeros, where
import numpy as np
import matplotlib.pyplot as plt
from logreg import LogisticRegression

if __name__ == "__main__":
    dosya_adi = 'data1.dat'
    veri = loadtxt(dosya_adi, delimiter=',')
    X = veri[:, 0:2]
    y = np.squeeze(np.array([veri[:, 2]]).T)
    n, d = X.shape

    ortalama = X.mean(axis=0)
    std_sapma = X.std(axis=0)
    X = (X - ortalama) / std_sapma

    logregModel = LogisticRegression(regLambda=0.00000001)
    logregModel.fit(X, y)

    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logregModel.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)

    plt.xlabel('Sinav 1 Skoru')
    plt.ylabel('Sinav 2 Skoru')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.savefig("grafik.png")
    # plt.show() calısmadı, bende proje dizininde foto olarak olusturtturdum.
    print("plt.show() çalismadi, bende proje dizininde foto olarak olusturtturdum (grafik.png).")