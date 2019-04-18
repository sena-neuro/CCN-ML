"""
    * Feature Extraction 
    (log power spectral density(PSD), magnitude squared spectral coherence, mutual info between electrode pairs, log ration of left-to-right hemisphere powers and anterior/posterior log power rations, all frequencies of interest ) <= Z-score normalization procedure

    * Feature selection 
    Candidate Feature >> useful features, Fisher discriminant ration(FDR), feature polling procedure.

    * Classification
    Mixture of Factor Analysis(MFA - based on maximum likelihood classification rule, allows rank ordering), Monte-Carlo cross validation, Kernelized principal component analysis (KPCA)
"""
import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np

def svc(x_train, y_train, x_test, y_test, gridsearch=True, verbose=False, kernel='rbf',
        gamma_grid=np.logspace(-15, 3, base=2, num=10), c_grid=np.logspace(-5, 15, base=2, num=10)):

    # coef grid, others?
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
            print("Best parameters set found on development set:")
            print()
            print(svc.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = svc.cv_results_['mean_test_score']

            stds = svc.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, svc.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()
    else:
        svc = SVC(kernel=kernel)
        svc.fit(x_train, y_train)
    if verbose:
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, svc.predict(x_test)
        print(metrics.classification_report(y_true, y_pred))
        print()

    accuracy = svc.score(x_test, y_test)
    return accuracy


'''
# to be used later
# more complicated but more useful
def improved_svm(X_train, y_train, X_test, y_test, multiclass = False):
    """
        Create the parameter grid and fit the information with the best model. 
        Return the parameters so that the usage of them will be easier.
    """

    param_grid = [
      {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']},
      {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': ["scale"], 'kernel': ['rbf']}
     ]
    grid_search = GridSearchCV(SVC(), param_grid, cv = 5)
    grid_search.fit(X_train, y_train)
    # to choose best params can use also the method: svc_param_selection()
    #print("    Best parameters: ",grid_search.best_params_)
    svm_model_linear = SVC(**grid_search.best_params_).fit(X_train, y_train) 
    svm_predictions = svm_model_linear.predict(X_test)    
    accuracy = svm_model_linear.score(X_test, y_test) 
    return accuracy

# NOT USED
def svc_param_selection(X, y, nfolds):
    """
        Method used to solve the grid problem.
        Not useful for the moment because of GridSearchCV library. 
        May be useful for a manual check. 
    """
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.predict()
    grid_search.best_params_
    return grid_search.best_params_

# Logistic Regression method
#---------------------------------------------------------------------------------
def logistic_reg(X_train1, y_train1, X_test1, y_test1):
    """
        Use this statistical method to find a classifier. 
    """
    try: 
        clf_D = LogisticRegression(solver='liblinear')
        clf_D.fit(X_train1, y_train1)
        predict = clf_D.predict(X_test1)
        score = [metrics.accuracy_score(y_test1, predict), metrics.precision_score(y_test1, predict),metrics.recall_score(y_test1, predict),metrics.f1_score(y_test1, predict)]
        return score
    except Exception: 
        print("    Something went wrong! improved_svm() method did not work!\n")
        


#-------------------------------------------------------
# Neural Network - Not finished 
# tensorflow to be added.
def simple_NN(x_train, x_test, y_train, y_test):
    """
        Simple Neural Network useful for the later work. 
    """

    model = Sequential()
    model.add(Dense(512, activation = 'relu', input_shape = (x.shape[1], )))
    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()
    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
def improved_NN(x_train, x_test, y_train, y_test):
    model2 = Sequential()
    model2.add(Dense(1000, activation = 'relu', input_shape = (x.shape[1], )))
    model2.add(Dropout(0.2))
    model2.add(Dense(1000, activation = 'relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(1000, activation = 'relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(1, activation = 'sigmoid'))
    model2.summary()
    model2.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model2.summary()

    checkpointer = ModelCheckpoint(filepath = 'MLP_new.weights.best.hdf5', verbose = 1, save_best_only = True)
    hist1 = model2.fit( x_train, y_train, epochs = 200, batch_size=512, 
                        validation_split = 0.1, callbacks = [checkpointer], verbose = 2, shuffle = True )
    score = model2.evaluate(x_test, y_test, verbose=1)
    print("Accuracy: ", score[1])
    predict3 = [1 if a>=0.5 else 0 for a in model2.predict(x_test)] 
    return

    # using checkpoints.
    checkpointer = ModelCheckpoint(filepath = 'MLP.weights.best.hdf5', verbose = 1, save_best_only = True)
    hist = model.fit(x_train, y_train, epochs = 100, batch_size=256, 
                     validation_split = 0.1, callbacks = [checkpointer], verbose = 2, shuffle = True)
    score = model.evaluate(x_test, y_test, verbose=1)
    print("Accuracy: ", score[1])
    predict2 = [1 if a>0.5 else 0 for a in model.predict(x_test)]
    return



#------------------------------------
# testing

if __name__ == "__main__":
    x_train = [[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0],
    		   [1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0]]
    x_test  = [[1,0,1,0,1],[1,0,1,0,1]]
    y_train = [1,0,1,0,1,0,1,0]
    y_test  = [1,1]
    
    print("Trying simple_svm()!\n")
    content = simple_svm(x_train,y_train, x_test, y_test)
    if content != None:
        print("    simple_svm() works.\n")
        
    print("Trying improved_svm()!\n")
    content = improved_svm(x_train,y_train, x_test, y_test)
    if content != None:
        print("    improved_svm() works.\n")
    
    print("Trying logistic_reg()!\n")
    content = logistic_reg(x_train,y_train, x_test, y_test)
    if content != None:
        print("    logistic_reg() works.\n")
'''
