#!/usr/bin/env python3

def sklearn_iris_to_df(sklearn_dataset):
    """
    processes the dataset from sklearn into a pandas dataframe with an
    additional column with string definition of the three classes.

    Input: (sklearn dataset)
    Output: (Pandas DataFrame Object)
    """
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df


def look_at_data(dataframe):
    """
    Simple function to generate scatter plot from a pandas dataframe
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['red', 'green', 'blue']

    for i, s in enumerate(dataframe.species.unique()):
        dataframe[dataframe.species==s].plot.scatter(x = 'sepal length (cm)',
                                            y = 'sepal width (cm)',
                                            label = s,
                                            color = colors[i],
                                            ax = ax)
    plt.show()


