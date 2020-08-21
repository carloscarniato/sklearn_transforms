from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
    
class CreateDummies(BaseEstimator, TransformerMixin):

    def __init__(self):
        print('in the CabinFeatureTransformer init method: ')
        
    def fit(self, x, y=None):        
        perfil_dummies = pd.get_dummies(x['PERFIL'], prefix='PERFIL')    
        self.perfil_columns=  perfil_dummies.columns
        return self

    def transform(self, x):
        # Retornamos um novo dataframe sem as colunas indesejadas
    
        perfil_dummies = pd.get_dummies(x['PERFIL'], prefix='PERFIL') 
        perfil_dummies = perfil_dummies.reindex(columns = self.perfil_columns, fill_value=0)
        
        x = pd.concat([x, perfil_dummies], axis=1)    

        return x
