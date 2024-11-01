import numpy as np

class LogisticRegression:

    def __init__(self, ogrenme_orani=0.01, duzenleme_katsayisi=0.01, epsilon=0.0001, max_iterasyon=10000):
        self.ogrenme_orani = ogrenme_orani
        self.duzenleme_katsayisi = duzenleme_katsayisi
        self.epsilon = epsilon
        self.max_iterasyon = max_iterasyon
        self.parametreler = None

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def maliyetHesapla(self, parametreler, X, y, duzenleme_katsayisi):
        m = len(y)
        tahmin = self.sigmoid(X @ parametreler)
        maliyet = (-1 / m) * (y.T @ np.log(tahmin) + (1 - y).T @ np.log(1 - tahmin))
        duzenleme_terimi = (duzenleme_katsayisi / (2 * m)) * np.sum(np.square(parametreler[1:]))
        return maliyet + duzenleme_terimi

    def gradyanHesapla(self, parametreler, X, y, duzenleme_katsayisi):
        m = len(y)
        tahmin = self.sigmoid(X @ parametreler)
        gradyan = (1 / m) * (X.T @ (tahmin - y))
        gradyan[1:] += (duzenleme_katsayisi / m) * parametreler[1:] 
        return gradyan

    def egit(self, X, y):
        n, d = X.shape
        X = np.c_[np.ones(n), X]
        self.parametreler = np.zeros(d + 1)

        for _ in range(self.max_iterasyon):
            gradyan = self.gradyanHesapla(self.parametreler, X, y, self.duzenleme_katsayisi)
            yeni_parametreler = self.parametreler - self.ogrenme_orani * gradyan

            if np.linalg.norm(yeni_parametreler - self.parametreler, ord=1) < self.epsilon:
                break
            self.parametreler = yeni_parametreler

    def tahmin(self, X):
        if self.parametreler is None:
            raise ValueError("Model egitilmedi. `egit` fonksiyonunu cagirin.")
        X = np.c_[np.ones(X.shape[0]), X]  # Sabit terim ekleniyor
        olasiliklar = self.sigmoid(X @ self.parametreler)
        return (olasiliklar >= 0.5).astype(int)
