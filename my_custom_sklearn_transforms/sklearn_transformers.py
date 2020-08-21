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

    def transform(self, X, **transformparams):
        dum = pd.get_dummies(X, columns=["PERFIL"]).copy()
        return pd.concat([X, dum], axis=1) 

    def fit(self, X, y=None, **fitparams):
        return self
