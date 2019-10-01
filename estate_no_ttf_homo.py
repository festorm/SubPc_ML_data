#!/home/koerstz/anaconda3/bin/python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split #in order to split the data set
from sklearn import preprocessing #scaling, transforming, and wrangling data.

from sklearn import svm
#these two are need to perform cross-validation
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import GridSearchCV

#metrics to model the performance
from sklearn.metrics import mean_squared_error, r2_score

#in order to save the model for future use
from sklearn.externals import joblib 

from multiprocessing import Pool
import sys

num_processes = int(sys.argv[1])

def training(seed):
    dssc = pd.read_pickle("estate_no_ttf.pkl")

    y = dssc.HOMO
    X = dssc

    for key in ['name', 'E_opt', 'HOMO', 'smiles'
               ]:
        X = X.drop(key,axis=1)


    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=0.2,
                                                random_state = seed)

    testing_size = [0.01,0.02,0.03,0.04,0.05,0.08,0.1,0.2,0.4,0.6,0.8,1.0]
    test_points = []
    r2_score_list = []
    rmse_list = []

    for i in testing_size:
        pipeline = make_pipeline(preprocessing.StandardScaler(),
                                         svm.SVR())
        hyperparameters = {'svr__C': [8,9,11,10,12,13,14],'svr__gamma':[0.0,0.01,0.001,0.02,0.03,0.005], 'svr__kernel': ['rbf','poly','sigmoid']}

        clf = GridSearchCV(pipeline, hyperparameters, cv=5)
        clf.fit(X_train[0:int(i*len(X_train))], y_train[0:int(i*len(y_train))]
    )
        pred = clf.predict(X_test)

        test_points.append(int(i*len(X_train)))
        r2_score_list.append(r2_score(y_test, pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, pred)))

    joblib.dump(clf, 'estate_nottf_homo_'+str(seed)+'.pkl')

    with open("training_seed_"+str(seed)+".csv","w") as f:
        f.write("{} \n ".format(",".join(map(str,test_points))))
        f.write("{} \n ".format(",".join(map(str,r2_score_list))))
        f.write("{} \n ".format(",".join(map(str,rmse_list))))

seeds = [15,10,30,42,80,90]
pool = Pool(processes=num_processes)
pool.map(training,seeds)
pool.close()
pool.join()

