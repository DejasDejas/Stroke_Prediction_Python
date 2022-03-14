from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import logging

log = logging.getLogger("__name__")


def up_sampling(X_train, y_train):
    """
    Create new instances from the current observations
    """
    smote: SMOTE = SMOTE(sampling_strategy='minority')  # create the  object with the desired sampling strategy
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)  # fit the object to our training data

    log.info("Classes counter before up sampling: \n{}".format(Counter(y_train)))
    log.info("Classes counter after up sampling : \n{}".format(Counter(y_train_balanced)))

    return X_train_balanced, y_train_balanced


def features_selections(model, X_train, y_train):
    """
    This function allows the selection of the variables that provide
    the most information  to the model given as parameters.
    """
    selector = SelectFromModel(model)
    selector.fit(X_train, y_train)
    log.info('Chosen columns : \n', X_train.columns[selector.get_support()])
    return selector


def normalization_data(X):
    """
    normalize data
    """
    scaler = StandardScaler()
    return scaler.fit(X)


class Treatment:
    """
    This class to effectuate data treatments.
    Params:
    - dataframe: pandas.DataFrame, raw data
    - target_feature: string, feature to predict name
    """

    def __init__(self, dataframe, target):
        """
        Attributes initialization
        """
        if target in dataframe.columns:
            self.dataframe = dataframe  # dataframe.copy()
            self.target = target
        else:
            raise ValueError("Target feature doesn't exist in dataframe.")

    def column_exist(self, column):
        """
        Function to check if a column exists.
        """
        return column in self.dataframe.columns

    def delete_column(self, column):
        """
        Function delete a column.
        params:
        - column: list, list of column to remove.
        """
        assert isinstance(column, list), "column variable must be a list !"
        for col in column:
            assert self.column_exist(col), "{} column doesn't exist.".format(column)
            self.dataframe.drop(col, axis=1, inplace=True)
        return

    def to_categorical(self):
        """
        This function allows a category to be modified by another category within a a categorical variable.
        """
        def label_encoding(label):
            from sklearn import preprocessing
            label_encoder = preprocessing.LabelEncoder()
            self.dataframe[label] = label_encoder.fit_transform(self.dataframe[label])
            self.dataframe[label].unique()
        categorical_columns = self.dataframe.select_dtypes('object').columns
        for column in categorical_columns:
            label_encoding(column)
        return

    def remove_outliers(self, column, replace=False):
        """
        This function remove outliers.
        """
        if self.column_exist(column):
            quantile_1 = self.dataframe[column].quantile(0.25)
            quantile_2 = self.dataframe[column].quantile(0.75)
            median = self.dataframe[column].median()
            inter_quantile = quantile_2 - quantile_1
            down_limit = quantile_1 - 1.5 * inter_quantile
            up_limit = quantile_2 + 1.5 * inter_quantile
            if replace:
                def replace_outliers(x): return median if (x < down_limit or x > up_limit) else x
                self.dataframe[column] = self.dataframe[column].apply(replace_outliers)
            else:
                self.dataframe = self.dataframe[(self.dataframe[column] > down_limit) &
                                                (self.dataframe[column] < up_limit)]
        else:
            raise ValueError(f"{column} column doesn't exist.")

    def split_dataset(self, dataframe):
        """
        This function split dataset into training data and test data
        """
        y = dataframe[self.target].values
        X = dataframe.drop(self.target, axis=1).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, shuffle=True)

        return X_train, X_test, y_train, y_test

    def encoding(self):
        """
        This function allows the encoding of qualitative values
        """
        dataframe = self.dataframe.copy()
        for col in dataframe.select_dtypes("category").columns:
            dataframe[col] = self.dataframe[col].cat.codes
        return dataframe

    def missing_values(self, column):
        """
        replace missing value by feature mean
        """
        self.dataframe[column] = self.dataframe[column].fillna(self.dataframe[column].mean())
        # fill 'bmi' NaN values using 'bmi' mean

        return self.dataframe
