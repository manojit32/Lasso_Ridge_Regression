import math
import numpy as np
import sys

def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

class l1_regularization():
    def __init__(self, alpha):
        self.alpha = alpha
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)
    def grad(self, w):
        return self.alpha * np.sign(w)

class l2_regularization():
    def __init__(self, alpha):
        self.alpha = alpha
    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w)
    def grad(self, w):
        return self.alpha * w

class Regression(object):
    def __init__(self, iters, lrate):
        self.iters = iters
        self.lrate = lrate
        self.training_errors = []
    def initialize_weights(self, n_features):
        limit = 1/math.sqrt(n_features)
        self.w = np.random.normal(-limit, limit, (n_features,))
    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.initialize_weights(n_features=X.shape[1])
        for i in range(self.iters):
            y_pred = X.dot(self.w)
            mse = np.mean(0.5*(y - y_pred)**2+self.regularization(self.w))
            self.training_errors.append(mse)
            grad_w = -(y - y_pred).dot(X)+self.regularization.grad(self.w)
            self.w = self.w - self.lrate * grad_w
    def predict(self, X):
        X = np.insert(X,0,1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred


class RidgeRegression(Regression):
    def __init__(self,alpha1, iters=3000, lrate=0.01):
        self.regularization = l2_regularization(alpha=alpha1)
        super(RidgeRegression, self).__init__(iters=iters,
                                                        lrate=lrate)
    def fit(self, X, y):
        X = normalize(X)
        super(RidgeRegression, self).fit(X, y)
    def predict(self, X):
        X = normalize(X)
        return super(RidgeRegression, self).predict(X)

class LassoRegression(Regression):
    def __init__(self, alpha1, iters=3000, lrate=0.01):
        self.regularization = l1_regularization(alpha=alpha1)
        super(LassoRegression, self).__init__(iters,
                                              lrate)
    def fit(self, X, y):
        X = normalize(X)
        super(LassoRegression, self).fit(X, y)
    def predict(self, X):
        X = normalize(X)
        return super(LassoRegression, self).predict(X)




def run():
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    df=pd.read_csv("AdmissionDataset/data.csv")
    X=df.iloc[:,0:8].as_matrix()
    #print(X.shape)
    y=df.iloc[:,8:9].values
    y=y.reshape((y.shape[0],))
    #print(y.shape)
    d_m=np.mean(X)
    d_s=np.std(X)
    d_n=(X-d_m)/d_s
    #print(d_n)
    #d_n=d_n.dropna(axis=1)
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(d_n,y)
    l=np.random.random(100)
    msel=[]
    for i in l:
        model =LassoRegression(i, iters=3000, lrate=0.001)
        #model =RidgeRegression(2,0.1, iters=3000, lrate=0.001)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_test, y_pred)
        msel.append(mse)
        #print("Mean squared error: %s" % (mse))
    plt.scatter(l,msel)
    plt.xlabel("alpha1")
    plt.ylabel("Error")
    plt.title("Lasso Regression")
    plt.show()
    msel=[]
    for i in l:
        #model =LassoRegression(2,i, iters=3000, lrate=0.001)
        model =RidgeRegression(i, iters=3000, lrate=0.001)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_test, y_pred)
        msel.append(mse)
        #print("Mean squared error: %s" % (mse))
    plt.scatter(l,msel)
    plt.xlabel("alpha1")
    plt.ylabel("Error")
    plt.title("Ridge Regression")
    plt.show()


def run2():
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    df=pd.read_csv("AdmissionDataset/data.csv")
    X=df.iloc[:,0:8].as_matrix()
    #print(X.shape)
    y=df.iloc[:,8:9].values
    y=y.reshape((y.shape[0],))
    #print(y.shape)
    d_m=np.mean(X)
    d_s=np.std(X)
    d_n=(X-d_m)/d_s
    from sklearn.model_selection import KFold
    fold=list(range(2,100))
    error=[]
    for i in fold:
        kf = KFold(n_splits=i)
        kf.get_n_splits(d_n)
        msel=[]
        for train_index, test_index in kf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = d_n[train_index], d_n[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model =RidgeRegression(i, iters=3000, lrate=0.001)
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(y_test, y_pred)
            msel.append(mse)
        error.append(min(msel))
    plt.scatter(fold,error)
    plt.xlabel("Value for K in K-Fold")
    plt.ylabel("Error")
    plt.title("Ridge Regression")
    plt.show()
    error=[]
    for i in fold:
        kf = KFold(n_splits=i)
        kf.get_n_splits(d_n)
        msel=[]
        for train_index, test_index in kf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = d_n[train_index], d_n[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model =LassoRegression(i, iters=100, lrate=0.001)
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(y_test, y_pred)
            msel.append(mse)
        error.append(min(msel))
    plt.scatter(fold,error)
    plt.xlabel("Value for K in K-Fold")
    plt.ylabel("Error")
    plt.title("Lasso Regression")
    plt.show()


def run3():
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    df=pd.read_csv("AdmissionDataset/data.csv")
    X=df.iloc[:,0:8].as_matrix()
    #print(X.shape)
    y=df.iloc[:,8:9].values
    y=y.reshape((y.shape[0],))
    #print(y.shape)
    d_m=np.mean(X)
    d_s=np.std(X)
    d_n=(X-d_m)/d_s
    from sklearn.model_selection import LeaveOneOut
    kf = LeaveOneOut()
    kf.get_n_splits(d_n)
    msel=[]
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = d_n[train_index], d_n[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model =RidgeRegression(0.0001, iters=3000, lrate=0.001)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        from sklearn.metrics import mean_squared_error,r2_score
        mse = mean_squared_error(y_test, y_pred)
        msel.append(mse)
    print("Mean Error for Ridge Regression : "+str(sum(msel)/len(msel)))
    msel=[]
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = d_n[train_index], d_n[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model =LassoRegression(0.0001, iters=3000, lrate=0.001)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_test, y_pred)
        msel.append(mse)
    print("Mean Error for Lasso Regression : "+str(sum(msel)/len(msel)))

def run4():
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    df=pd.read_csv("AdmissionDataset/data.csv")
    X=df.iloc[:,0:8].as_matrix()
    #print(X.shape)
    y=df.iloc[:,8:9].values
    y=y.reshape((y.shape[0],))
    #print(y.shape)
    d_m=np.mean(X)
    d_s=np.std(X)
    d_n=(X-d_m)/d_s
    #print(d_n)
    #d_n=d_n.dropna(axis=1)
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(d_n,y)
    l=np.random.random(100)
    msel=[]
    mset=[]
    for i in l:
        model =LassoRegression(i, iters=3000, lrate=0.001)
        #model =RidgeRegression(0.1, iters=3000, lrate=0.001)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        y_pred2=model.predict(X_train)
        from sklearn.metrics import mean_squared_error,r2_score
        mse = r2_score(y_test, y_pred)
        mse2 = r2_score(y_train, y_pred2)
        msel.append(mse)
        mset.append(mse2)
        #print("Mean squared error: %s" % (mse))
    plt.scatter(l,msel)
    plt.scatter(l,mset)
    plt.xlabel("alpha1")
    plt.ylabel("R2 Score")
    plt.title("Lasso Regression")
    plt.show()
    msel=[]
    mset=[]
    for i in l:
        #model =LassoRegression(i, iters=3000, lrate=0.001)
        model =RidgeRegression(i, iters=3000, lrate=0.001)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        y_pred2=model.predict(X_train)
        from sklearn.metrics import mean_squared_error,r2_score
        mse = r2_score(y_test, y_pred)
        mse2 = r2_score(y_train, y_pred2)
        msel.append(mse)
        mset.append(mse2)
        #print("Mean squared error: %s" % (mse))
    plt.scatter(l,msel)
    plt.scatter(l,mset)
    plt.xlabel("alpha1")
    plt.ylabel("R2 Score")
    plt.title("Ridge Regression")
    plt.show()


def run5():
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    df=pd.read_csv("AdmissionDataset/data.csv")
    X=df.iloc[:,0:8].as_matrix()
    #print(X.shape)
    y=df.iloc[:,8:9].values
    y=y.reshape((y.shape[0],))
    #print(y.shape)
    d_m=np.mean(X)
    d_s=np.std(X)
    d_n=(X-d_m)/d_s
    #print(d_n)
    #d_n=d_n.dropna(axis=1)
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(d_n,y)
    
    model =LassoRegression(0.000001, iters=3000, lrate=0.001)
    #model =RidgeRegression(0.1, iters=3000, lrate=0.001)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    print(model.w)
    print("Mean Squared Error : "+str(mse))
    plt.figure(figsize=(10,6))
    plt.plot(df.columns,model.w,"o-")
    plt.xlabel("Weights")
    plt.ylabel("Values")
    plt.title("Lasso Regression")
    plt.grid(True)
    plt.show()
    
    #model =LassoRegression(i, iters=3000, lrate=0.001)
    model =RidgeRegression(0.000001, iters=3000, lrate=0.001)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    print(model.w)
    print("Mean Squared Error : "+str(mse))
    plt.figure(figsize=(10,6))

    plt.plot(df.columns,model.w,"o-")
    plt.xlabel("Weights")
    plt.ylabel("Values")
    plt.title("Lasso Regression")
    plt.grid(True)
    plt.show()
    