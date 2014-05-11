import numpy as num
from numpy.ma import vstack
from sklearn import linear_model, metrics, cross_validation
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from helpers.config_helpers import parse_config
from helpers.io_helpers import search_files_by_extension, get_absolute_path

__author__ = 'Daeyun Shin'


def training(training_data_dir):
    paths = search_files_by_extension(training_data_dir, ["txt"])

    X = None
    Y = None

    for path in paths:
        data = num.genfromtxt(path, delimiter=',')

        x = data[:, 0].astype(num.float32)
        y = data[:, 1:].astype(num.float32)
        if X == None or Y == None:
            X = x
            Y = y
        else:
            X = vstack(X, x)
            Y = vstack(Y, y)

    print X.shape, Y.shape

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)

    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    rbm.n_components = 100
    logistic.C = 6000.0

    classifier.fit(X_train, Y_train)

    print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            classifier.predict(X_test))))

    n_samples = Y.shape[0]
    cv = cross_validation.StratifiedShuffleSplit(Y, n_iter=3, test_size=0.25, random_state=0)
    result = cross_validation.cross_val_score(classifier, X, Y, cv=cv)
    print result

