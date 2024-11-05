import numpy as np

class LogisticRegression:

    def __init__(self, alpha=0.01, regLambda=0.01, epsilon=0.0001, maxNumIters=10000):
        self.parametreler = None

    def computeCost(self, theta, X, y, regLambda):
        m = len(y)
        tahmin = self.sigmoid(X @ theta)
        maliyet = (-1 / m) * (y.T @ np.log(tahmin) + (1 - y).T @ np.log(1 - tahmin))
        duzenleme_terimi = (regLambda / (2 * m)) * np.sum(np.square(theta[1:]))
        return maliyet + duzenleme_terimi

    def computeGradient(self, theta, X, y, regLambda):
        m = len(y)
        tahmin = self.sigmoid(X @ theta)
        gradyan = (1 / m) * (X.T @ (tahmin - y))
        gradyan[1:] += (regLambda / m) * theta[1:]
        return gradyan

    def fit(self, X, y):
        n, d = X.shape
        X = np.concatenate((np.ones((n, 1)), X), axis=1)
        self.parametreler = np.zeros(d + 1)

        for a in range(self.maxNumIters):
            gradyan = self.computeGradient(self.parametreler, X, y, self.regLambda)
            yeni_parametreler = self.parametreler - self.alpha * gradyan

            if np.linalg.norm(yeni_parametreler - self.parametreler, ord=1) < self.epsilon:
                break
            self.parametreler = yeni_parametreler

    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        olasiliklar = self.sigmoid(X @ self.parametreler)
        return (olasiliklar >= 0.5).astype(int)
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
