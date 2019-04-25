#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import train_test_split

def generate_stats_data(x,y):
    """
        Trying to generate data to check the code with different distributions of data. 
    """

    try: 
        X_columns = ['mean', 'standard deviation', 'kurt', 'skewness']
        Y_columns = ['label']

        X = pd.DataFrame(columns = X_columns)
        Y = pd.DataFrame(columns = Y_columns)

        for i in range(len(x)):
            X.loc[i] = np.array([np.mean(x[i]), np.std(x[i]), scipy.stats.kurtosis(x[i]), scipy.stats.skew(x[i])])
            Y.loc[i] = y[i]

        # Split the 'features' and 'income' data into training and testing sets
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y,test_size = 0.2, random_state = 0)

        return X_train1, X_test1, y_train1, y_test1 
    except Exception: 
        print("    Something went wrong with generate_stats_data()")


#------------------------------------
# testing
if __name__ == "__main__":

    x = [[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0],
         [1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0]]
    y = [1,0,1,0,1,0,1,0]
    print("Trying generate_stats_data()!\n")
    content    = generate_stats_data(x,y)
    if content != None:
        print("    generate_stats_data() works.\n")

