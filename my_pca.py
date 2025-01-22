import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class MyPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.log_likelihood = None

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

        print('------------------------------------------------------->>',idx[0])
        print('Meu autovetor da posição do meu maior autovalor:', idx, ' - ', eigenvectors[:, idx[0]])
        print('###')
        print(np.array(eigenvectors[:, idx[0]]))
        print(np.array(eigenvectors[:, idx[0]]).reshape(-1, 1))
        print('###')

        # idx_vectors = np.argsort(eigenvectors)[::-1]
        # eigenvectors = eigenvectors[idx_vectors]
        eigenvectors = eigenvectors[:, idx[0:1]]

        # print('3.5-', eigenvectors)
        # eigenvectors = eigenvectors[:, -1]

        print('3-\n AutoValores: \n', self.eigenvalues, '\nAutoVetores:\n ', eigenvectors)

        self.components = eigenvectors.T

        # print('\n', self.components, '\n')

    def transform(self, X):
        # print('\n------------ Transform ------------\n')
        # print('\n X orig: \n', X)

        X = X - self.mean

        # print('\n X alter: \n', X)

        # print('\n------------ Cálculo Final ------------\n')
        # print(X)
        # print(self.components.T)

        # print('\n Retorno: \n', np.dot(X, self.components.T))

        return np.dot(X, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def plot_scree(self):
        plt.figure(figsize=(10,6))
        plt.title('Scree Plot')
        plt.xlabel('Components')
        plt.ylabel('Eigenvalue')
        plt.grid(True)

        plt.plot(range(1, len(self.eigenvalues) + 1), self.eigenvalues, 'o-', linewidth=2, color='blue')

        plt.show()

    def plot_explained_variance(self):
        total_variance = np.sum(self.eigenvalues)
        var_exp= [(i/total_variance) for i in sorted(self.eigenvalues, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)

        plt.figure(figsize=(10,6))
        plt.title('Explained Variance')
        plt.xlabel('Components')
        plt.ylabel('Variance')
        plt.grid(True)

        plt.plot(range(1, len(self.eigenvalues) + 1), cum_var_exp)

        plt.show()

    def calculate_profile_likelihood(self):
        L_max = len(self.eigenvalues)
        log_likelihoods = []
        epsilon = 1e-10

        for L in range(0, L_max):
            mu1 = np.mean(self.eigenvalues[:L])
            mu2 = np.mean(self.eigenvalues[L:])
            var = ( np.sum((self.eigenvalues[:L] - mu1) ** 2) +  np.sum((self.eigenvalues[L:] - mu2) ** 2) ) / L_max + epsilon

            ll1 = np.sum(norm.logpdf(self.eigenvalues[:L], loc=mu1, scale=np.sqrt(var)))
            ll2 = np.sum(norm.logpdf(self.eigenvalues[L:], loc=mu2, scale=np.sqrt(var)))
            total_ll = ll1 + ll2

            log_likelihoods.append(total_ll)
            self.log_likelihood = log_likelihoods

        return np.argmax(log_likelihoods) + 1

    def plot_profile_likelihood(self):
        if self.log_likelihood is None:
            self.calculate_profile_likelihood()

        plt.figure(figsize=(10,6))
        plt.title('Profile Likelihood')
        plt.xlabel('Components')
        plt.ylabel('Likelihood')
        plt.grid(True)

        plt.plot(range(1, len(self.log_likelihood) + 1), self.log_likelihood, 'o-', linewidth=2, color='blue')

        plt.show()
