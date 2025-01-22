from my_pca import MyPCA

pca = MyPCA(n_components=2)

X = [
        [14, 1, 1, 50000],
        [22, 0, 0, 12000],
        [ 7, 1, 1, 35000],
        [15, 1, 1, 65000],

        # [13, 0, 1, 50000],
        # [22, 0, 0, 11000],
        # [ 6, 1, 1, 36000],
        # [17, 0, 1, 67000],

        # [16, 1, 0, 51000],
        # [23, 1, 1, 13000],
        # [ 6, 0, 0, 34000],
        # [13, 1, 0, 67000],
    ]

x_pca = pca.fit_transform(X)

print('\n\n\n My PCA: ', x_pca)
print(pca.transform([[7, 1, 0, 25000]]))

pca.plot_scree()
pca.plot_explained_variance()

print('Número ótimo de componentes:', pca.calculate_profile_likelihood())
pca.plot_profile_likelihood()


# from sklearn.decomposition import PCA

# sk_pca = PCA(n_components=2)

# res = sk_pca.fit_transform(X)

# print('SkLearn:', res)
