"""
    * Feature Extraction 
    (log power spectral density(PSD), magnitude squared spectral coherence, mutual info between electrode pairs, log ration of left-to-right hemisphere powers and anterior/posterior log power rations, all frequencies of interest ) <= Z-score normalization procedure

    * Feature selection 
    Candidate Feature >> useful features, Fisher discriminant ration(FDR), feature polling procedure.

    * Classification
    Mixture of Factor Analysis(MFA - based on maximum likelihood classification rule, allows rank ordering), Monte-Carlo cross validation, Kernelized principal component analysis (KPCA)
"""

# svm is implemented using libsvm
import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model  import LogisticRegression
#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras.callbacks import ModelCheckpoint

# SVM 
#---------------------------------------------------------------------------------
# useful for now
def simple_svm(x_train,y_train, x_test, y_test):
	try:
		svm_model_linear = SVC(kernel = 'linear', gamma='auto').fit(x_train, y_train) 
		y_pred   = svm_model_linear.predict(x_test)    
		accuracy = svm_model_linear.score(x_test, y_test) 
		from sklearn.metrics import classification_report, confusion_matrix  
		print(confusion_matrix(y_test,y_pred))  
		print(classification_report(y_test,y_pred)) 
		return accuracy
	except Exception: 
		print("    Something went wrong! simple_svm() method did not work!\n")


# to be used later
# more complicated but more useful
def improved_svm(X_train, y_train, X_test, y_test, multiclass = False):
    """
        Create the parameter grid and fit the information with the best model. 
        Return the parameters so that the usage of them will be easier.
    """
    try: 
        param_grid = [
          {'C': [0.1,0.7,0.8,0.9,1], 'kernel': ['linear']},
          {'C': [0.9,1, 1.1], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
         ]
        grid_search = GridSearchCV(SVC(), param_grid, cv = 2)
        grid_search.fit(X_train, y_train)
        # to choose best params can use also the method: svc_param_selection()
        #print("    Best parameters: ",grid_search.best_params_)
        svm_model_linear = SVC(grid_search.best_params_).fit(X_train, y_train) 
        svm_predictions = svm_model_linear.predict(X_test)    
        accuracy = svm_model_linear.score(X_test, y_test) 
        return accuracy
    except Exception:
        print("    Something went wrong! improved_svm() method did not work!\n")

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
        

'''
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
'''


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
