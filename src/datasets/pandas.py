import pandas as pd

from .dataset import Dataset

class PandasDataset(Dataset):
    def __init__(self, name, df, ignore = None):
        self.name = name

        self.X = df
        if ignore != None:
            for column in ignore:
                self.X = self.X.drop(column, axis = 1)

        self.length = len(self.X)
        y = pd.np.zeros(self.length)
        X_train = self.X
        y_train = y
        X_test = self.X
        y_test = y
        self.result = df
        self._data = X_train, y_train, X_test, y_test

    def data(self):
        return self._data