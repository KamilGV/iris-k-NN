from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


def find_classes_sklearn_knn(training_table: pd.DataFrame, sample_table: pd.DataFrame, k_neighbours=5) -> pd.DataFrame:
    knn = KNeighborsClassifier(n_neighbors=k_neighbours)

    data = training_table.drop('Species', axis=1)
    target = training_table['Species'].to_numpy()
    knn.fit(data, target)

    sample = sample_table.iloc[:, :4].to_numpy()
    predict = knn.predict(sample)
    sample_table.loc[:, 'Species'] = predict
    return sample_table



