import pandas as pd
import numpy as np
from math import sqrt


def find_classes_my_knn(training_table: pd.DataFrame, sample_table: pd.DataFrame, k_neighbours=5) -> pd.DataFrame:
    for index in sample_table.index:
        table = training_table.copy()
        table['Distance'] = pow((pow(table['SepalLengthCm'] - sample_table.loc[index, 'SepalLengthCm'], 2) +
                                 pow(table['SepalWidthCm'] - sample_table.loc[index, 'SepalWidthCm'], 2) +
                                 pow(table['PetalLengthCm'] - sample_table.loc[index, 'PetalLengthCm'], 2) +
                                 pow(table['PetalWidthCm'] - sample_table.loc[index, 'PetalWidthCm'], 2)), 1 / 2)
        table.sort_values(by=['Distance'], ascending=True, inplace=True)
        table = table.iloc[:k_neighbours, :]
        sample_table.loc[index, 'Species'] = table['Species'].mode()[0]
    return sample_table
