import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from my_knn import find_classes_my_knn
from sklearn_knn import find_classes_sklearn_knn

COUNT_SAMPLE = 10
K_NEIGHBOURS = 5
new_table = pd.read_csv('Iris.csv', index_col=0)

sample_selected_indices = np.random.choice(new_table.index, COUNT_SAMPLE, replace=False)
sample_table = new_table.loc[sample_selected_indices]
sample_table_with_answers = sample_table.copy()
sample_table.loc[:, 'Species'] = np.nan
training_table = new_table.drop(sample_selected_indices)


my_table = find_classes_my_knn(training_table, sample_table.copy(), K_NEIGHBOURS)
sample_table_with_answers['MySpecies'] = my_table['Species']
sample_table_with_answers['myAnswersIsCorrect'] = (
        sample_table_with_answers['Species'] == (sample_table_with_answers['MySpecies']))


sklearn_table = find_classes_sklearn_knn(training_table, sample_table.copy(), K_NEIGHBOURS)
sample_table_with_answers['sklearnSpecies'] = my_table['Species']
sample_table_with_answers['sklearnAnswersIsCorrect'] = (
        sample_table_with_answers['Species'] == (sample_table_with_answers['sklearnSpecies']))

print(sample_table_with_answers.iloc[:, [4, 5, 7]])
print('')
print("My K-NN Correct: ",
      100*(sample_table_with_answers['myAnswersIsCorrect'].value_counts().iloc[0]/COUNT_SAMPLE), "%")
print("Sklearn K-NN Correct: ",
      100*(sample_table_with_answers['sklearnAnswersIsCorrect'].value_counts().iloc[0]/COUNT_SAMPLE), "%")




