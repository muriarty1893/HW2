import numpy as np

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.parametreler = None

    def computeCost(self, parametreler, veri, etiket, regLambda):
        ornek_sayisi = len(etiket)
        tahminler = self.sigmoid(veri @ parametreler)
        maliyet = (-1 / ornek_sayisi) * (etiket.T @ np.log(tahminler) + (1 - etiket).T @ np.log(1 - tahminler))
        duzenleme_terimi = (regLambda / (2 * ornek_sayisi)) * np.sum(np.square(parametreler[1:]))
        return maliyet + duzenleme_terimi

    def computeGradient(self, parametreler, veri, etiket, regLambda):
        ornek_sayisi = len(etiket)
        tahminler = self.sigmoid(veri @ parametreler)
        gradyan = (1 / ornek_sayisi) * (veri.T @ (tahminler - etiket))
        gradyan[1:] += (regLambda / ornek_sayisi) * parametreler[1:]
        return gradyan

    def fit(self, veri, etiket):
        ornek_sayisi, ozellik_sayisi = veri.shape
        veri = np.concatenate([np.ones((ornek_sayisi, 1)), veri], axis=1)
        self.parametreler = np.zeros(ozellik_sayisi + 1)

        for iterasyon in range(self.maxNumIters):
            gradyan = self.computeGradient(self.parametreler, veri, etiket, self.regLambda)
            yeni_parametreler = self.parametreler - self.alpha * gradyan

            # Convergence kontrolü
            norm_degisim = np.linalg.norm(yeni_parametreler - self.parametreler, ord=2)
            if norm_degisim <= self.epsilon:
                print(f'Convergence reached at iteration {iterasyon}, Change in theta: {norm_degisim:.6f}')
                break

            self.parametreler = yeni_parametreler

        # Son iterasyondaki değişimi yazdır
        if norm_degisim > self.epsilon:
            print(f'Max iterations reached without full convergence. Final change in theta: {norm_degisim:.6f}')


    def predict(self, veri):
        if self.parametreler is None:
            raise ValueError("Model is not trained yet. Call `fit` before `predict`.")
        veri = np.concatenate([np.ones((veri.shape[0], 1)), veri], axis=1)
        olasiliklar = self.sigmoid(veri @ self.parametreler)
        return (olasiliklar >= 0.5).astype(int)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
