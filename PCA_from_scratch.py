# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 22:32:11 2022

@author: peria
"""

import pandas as pd
import numpy as np

class PCA_from_scratch:
    
    def __init__(self, k_components):
        # number of principal components to be returned
        self.k_components = k_components
        # principal directions are eigenvectors of the covariance matrix
        self.principal_directions = None
        # mean to be subtracted from X in order to center the problem and express Cov(X) = (1/n)*Xt @ X
        self.mean = None
        # Variance associated to every principal component
        self.variance_along_principal_direction = None
        # Proportion of explained variance associated with every principal component
        self.explained_variance_ratio = None
        # Cumulative roportion of explained variance
        self.cum_explained_variance_ratio = None
    
    def fit(self, X):
        # Factors are in cols, so I do the mean column wise
        self.mean = np.mean(X, axis = 0)
        X = X - self.mean
        
        # rowvar = False because in X I want observations on rows and factors on cols
        covariance_matrix = np.cov(X, rowvar = False)
        
        # eigenvalues and eigenvectors aren't necessary ordered
        # eigenvectors are column vectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # Now I sort eigenvalues and eigenvectors from the biggest --> argsort returns the indices that sort an array
        idx_eigenvalues_ordered = np.argsort(eigenvalues)[::-1] # with -1 I read from the end of the array
        eigenvalues_ordered = eigenvalues[idx_eigenvalues_ordered]
        eigenvectors_ordered = eigenvectors.T[idx_eigenvalues_ordered] # eigenvectors_ordered are on rows now
        
        self.explained_variance_ratio = eigenvalues_ordered / np.sum(eigenvalues_ordered)
        
        self.cum_explained_variance_ratio = np.cumsum(eigenvalues_ordered) / np.sum(eigenvalues_ordered)
        
        # If I want the first k-PCs, I have to consider only the first k-principal_directions (aka eigenvectors of covariance)
        self.principal_directions = eigenvectors_ordered[0:self.k_components,:]
        
        # If I want the vaiance of data along the first k principal directions, I have to consider only the first k-eigenvalues
        self.variance_along_principal_direction = eigenvalues_ordered[0:self.k_components]
        
        # If I want the explained variance ratio for first k-PCs, I have to consider only the first k-principal_directions
        self.explained_variance_ratio = self.explained_variance_ratio[0:self.k_components]
        self.cum_explained_variance_ratio = self.cum_explained_variance_ratio[0:self.k_components]

        
    def transform(self, X):
        X = X - self.mean
        principal_components = X @ self.principal_directions.T
        return(principal_components)