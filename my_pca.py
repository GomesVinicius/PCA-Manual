import numpy as np

class MyPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None

    def fit(self, X):
        print('Base orig: \n', X)

        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        print('Média: \n', self.mean, ' - ', 'Base centrada na média: \n', X)

        covariance_matrix = np.cov(X, rowvar=False)
        print('Matriz de Covariância: \n', covariance_matrix)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        print('1-\n AutoValores: \n', eigenvalues, '\nAutoVetores:\n ', eigenvectors)
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        print('2-\n AutoValores: \n', self.eigenvalues, '\nAutoVetores:\n ', eigenvectors)

        eigenvectors = eigenvectors[:, :self.n_components]

        print('3-\n AutoValores: \n', self.eigenvalues, '\nAutoVetores:\n ', eigenvectors)

        self.components = eigenvectors.T

        print('\n', self.components, '\n')

    def transform(self, X):
        print('\n------------ Transform ------------\n')
        print('\n X orig: \n', X)

        X = X - self.mean

        print('\n X alter: \n', X)

        print('\n------------ Cálculo Final ------------\n')
        print(X)
        print(self.components.T)

        print('\n Retorno: \n', np.dot(X, self.components.T))

        return np.dot(X, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
