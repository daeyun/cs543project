import numpy as num
from sklearn import linear_model, metrics, cross_validation
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from helpers.config_helpers import parse_config
from helpers.io_helpers import search_files_by_extension, get_absolute_path
from sklearn.externals import joblib
from helpers.io_helpers import get_absolute_path, search_files_by_extension, pretty_print_exception, \
    make_sure_dir_exists
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

__author__ = 'Daeyun Shin'


def training(training_data_dir, num_instances=None, instance_id=None):
    paths = search_files_by_extension(training_data_dir, ["txt"])

    X = None
    Y = None

    #for path in paths:
        #data = num.genfromtxt(path, delimiter=',')

        #if data.ndim == 1:
            #data = num.array([data])

        #x = data[:, 1:].astype(num.float32)
        #y = data[:, 0].astype(num.float32)
        #y.resize((y.shape[0], 1))

        #if y[0,0]==0:
            #n = x.shape[0]
            #p = num.random.permutation(n)
            #x=x[p[:n/15], :]
            #y=y[p[:n/15], :]

        #if X == None or Y == None:
            #X = x
            #Y = y
        #else:
            #X = num.vstack((X, x))
            #Y = num.vstack((Y, y))
        #print 'loaded {}'.format(path)
    #num.savetxt('training_data_X.txt', X, fmt="%12.10G")
    #num.savetxt('training_data_Y.txt', Y, fmt="%12.10G")

    X = num.genfromtxt('training_data_X.txt', unpack=True)
    Y = num.genfromtxt('training_data_Y.txt', unpack=True)
    X = num.transpose(X)
    #Y.resize((Y.shape[0], 1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=89)

    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)

    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    if instance_id is not None:
        r, n_comp, c = [
                (0.0001, 200, 500),
                (0.0005, 200, 500),
                (0.0006, 200, 500),
                (0.0007, 200, 500),
                (0.0010, 200, 500),
                (0.0015, 200, 500),
                (0.0020, 200, 500),
                (0.0025, 200, 500),
                (0.0030, 200, 500),
                (0.0040, 1000, 500),

                (0.0001, 1000, 500),
                (0.0005, 1000, 500),
                (0.0006, 1000, 500),
                (0.0007, 1000, 500),
                (0.0010, 1000, 500),
                (0.0015, 1000, 500),
                (0.0020, 1000, 500),
                (0.0025, 1000, 500),
                (0.0030, 1000, 500),
                (0.0040, 1000, 500),
                ][instance_id]
    else:
        r, n_comp, c = (0.02, 500, 4000)

    print r, n_comp, c
    rbm.learning_rate = r
    rbm.n_iter = 40
    rbm.n_components = n_comp
    logistic.C = c
    #rbm.learning_rate = 0.02
    #rbm.n_iter = 20
    #rbm.n_components = 200
    #logistic.C = 6000

    #classifier.fit(X_train, Y_train)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    #neigh = KNeighborsClassifier(n_neighbors=5)
    #classifier = DecisionTreeClassifier(max_depth=5)
    #classifier = AdaBoostClassifier()
    classifier = SVC(gamma=10, C=10)
    #classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    classifier.fit(X_train, Y_train) 
    print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            classifier.predict(X_test))))



    exit()





    n_samples = Y.shape[0]
    cv = cross_validation.StratifiedShuffleSplit(Y, n_iter=20, test_size=0.3, random_state=9)
    result = cross_validation.cross_val_score(classifier, X, Y, cv=cv)
    print "Learning rate: {}, N components: {}, logistic C: {}".format(r, n_comp, c)
    print result
    print "Result: {}".format(num.mean(result))

    out_dir = '/home/ubuntu/mount/cs543/out/rr/'
    make_sure_dir_exists(out_dir)
    filename = '{}_{}_{}_{}.clf'.format(instance_id, r, n_comp, c)
    full_path = os.path.join(out_dir, filename)
    _ = joblib.dump(classifier, full_path, compress=9)
    print 'saved classifier as {}'.format(full_path)
