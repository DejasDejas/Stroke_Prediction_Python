from src.data.treatment import Treatment
from src.data.make_naive_dataset import load_dataset, get_dataset_folder_names
import time
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import logging
from src.visualization.visualize import plot_acc_resume_multi_model, plot_roc_curve
import pickle
from config.definitions import ROOT_DIR
import os

log = logging.getLogger("__name__")


def preprocessing_data(dataframe, target_feature, verbose=True):
    """
    split and prepare data for training model
    """
    treatment = Treatment(dataframe, target_feature)
    X_tr, X_te, y_tr, y_te = treatment.split_dataset(dataframe)
    if verbose:
        log.info('Samples in Train Set:', len(X_tr))
        log.info('Samples in Test Set:', len(X_te))
    return X_tr, X_te, y_tr, y_te


def timeit(method):
    """
    python decorator to measure the execution time of methods:
    """

    def timed(*args, **kw):
        time_start = time.time()
        result = method(*args, **kw)
        time_end = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((time_end - time_start) * 1000)
        else:
            log.info('{} {:.2f} ms'.format(method.__name__, (time_end - time_start) * 1000))
        return result

    return timed


@timeit
def classifier_training(current_model, X, y):
    current_model.fit(X, y)


def calcul_sensitivity_specificity(y_true, y_p, verbose=True):
    conf_matrix = confusion_matrix(y_true, y_p)
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    # calculate the sensitivity
    sensitivity = TP / (TP + FN)
    # calculate the specificity
    specificity = TN / (TN + FP)
    if verbose:
        log.info("Sensitivity score for SVM: {:.2f}".format(sensitivity))
        log.info(("Specificity score for SVM: {:.2f}".format(specificity)))
    return sensitivity, specificity


def compute_score(current_model, x_t, y_t, verbose=True):
    y_p = current_model.predict(x_t)
    y_p_proba = current_model.predict_proba(x_t)
    score = accuracy_score(y_t, y_p)
    if verbose:
        log.info("Accuracy score for {}: {:.2f}%".format(type(current_model).__name__, score * 100))
    return y_p, y_p_proba, score


# define the function blocks
def svm():
    """
    SVM model
    """
    log.info('Define SVM model')
    return SVC(kernel='rbf', random_state=0, probability=True)


def logistic_regression():
    """
    Logistic regression model
    """
    log.info('Define Logistic regression model')
    return LogisticRegression(random_state=0)


def knn():
    """
    KNN model
    """
    log.info('Define KNN model')
    return KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)


def gaussian():
    """
    Gaussian model
    """
    log.info('Define Gaussian model')
    return GaussianNB()


def decision_tree():
    """
    Decision tree model
    """
    log.info('Define Decision tree model')
    return DecisionTreeClassifier(criterion='entropy', random_state=0)


def lgbm():
    """
    LGBM model
    """
    log.info('Define LGBM model')
    return LGBMClassifier(n_estimators=100, random_state=42)


def xgbc():
    """
    XGBC model
    """
    log.info('Define XGBC model')
    return XGBClassifier(objective="binary:logistic", random_state=42)


def random_forest():
    """
    Random forest model
    """
    log.info('Define Random forest model')
    return RandomForestClassifier(n_estimators=100, random_state=42)


# map the inputs to the function blocks
build_model = {'svm': svm,
               'lr': logistic_regression,
               'knn': knn,
               'gaussian': gaussian,
               'dt': decision_tree,
               'lgbm': lgbm,
               'xgbc': xgbc,
               'rf': random_forest
               }

if __name__ == '__main__':

    # load naive made dataset
    _, dataset_name, raw_data_path, processed_data_path = get_dataset_folder_names()

    target = 'stroke'
    data = load_dataset(raw_data_path, processed_data_path, dataset_name)
    X_train, X_test, y_train, y_test = preprocessing_data(data, target)

    # define model:
    scores = {}
    model_name = ['svm', 'lr', 'knn', 'gaussian', 'dt', 'lgbm', 'xgbc', 'rf']
    for model_type in model_name:
        model = build_model[model_type]()  # select model
        classifier_training(model, X_train, y_train)  # training model
        y_pred, y_pred_proba, acc = compute_score(model, X_test, y_test)  # estimate prediction and compute acc score
        _, _ = calcul_sensitivity_specificity(y_test, y_pred)  # sensitivity and specificity scores
        scores[model_type] = acc
        # plot roc curve for each model:
        plot_roc_curve(model_type, y_test, y_pred_proba, save_figure=True)
        save_path = os.path.join(ROOT_DIR, 'models', model_type + '.sav')  # define model name for saving
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)  # to load this model: "loaded_model = pickle.load(open(save_path, 'rb'))"

    # plot accuracy score resume for all selected model:
    plot_acc_resume_multi_model(scores, save_figure=True)
