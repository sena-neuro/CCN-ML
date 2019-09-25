"""
    * Feature Extraction 
    (log power spectral density(PSD), magnitude squared spectral coherence, mutual info between electrode pairs, log ration of left-to-right hemisphere powers and anterior/posterior log power rations, all frequencies of interest ) <= Z-score normalization procedure

    * Feature selection 
    Candidate Feature >> useful features, Fisher discriminant ration(FDR), feature polling procedure.

    * Classification
    Mixture of Factor Analysis(MFA - based on maximum likelihood classification rule, allows rank ordering), Monte-Carlo cross validation, Kernelized principal component analysis (KPCA)
"""
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def svc(x_train, y_train, x_test, y_test, gridsearch=True, verbose=False, kernel='rbf',
        gamma_grid=np.logspace(-15, 3, base=2, num=10), c_grid=np.logspace(-5, 15, base=2, num=10)):

    if gridsearch:
        svc = SVC(kernel=kernel)  # check
        if kernel == 'rbf':
            svc = GridSearchCV(svc, cv=5,n_jobs=-1,
                               param_grid={"C": c_grid,
                                           "gamma": gamma_grid})
        else:
            print("Gridsearch for degree and coef is not implemented, only the optimal gamma value will be searched")
            svc = GridSearchCV(svc, cv=5,n_jobs=-1,
                               param_grid={"gamma": gamma_grid})
        svc.fit(x_train, y_train)
        if verbose:
            print("kernel: ", kernel)
            print("Best parameters set found on development set:\n")
            print(svc.best_params_ )
            print("\nGrid scores on development set:\n")

            means = svc.cv_results_['mean_test_score']
            stds = svc.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, svc.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
    else:
        svc = SVC(kernel=kernel)
        svc.fit(x_train, y_train)

    y_pred = svc.predict(x_test)
    if verbose:
        print("Detailed classification report: \n")
        print("The model is trained on the full development set.\n")
        print("The scores are computed on the full evaluation set.\n")
        print(metrics.classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred