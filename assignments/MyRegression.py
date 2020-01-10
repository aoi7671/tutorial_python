

class MyLinearRegression(BaseEstimator, RegressorMixin):

    def __init__(self, lam = 0,a=0,b=0):
        """
        Initialize a coefficient and an intercept.
        """
        self.a = a
        self.b = b
        self.lam = lam
        
    def fit(self, X, y):
        """
        X: array-like, shape (n_samples, n_features)
        y: array, shape (n_samples,)
        Estimate a coefficient and an intercept　from data.
        """
        X, y = check_X_y(X, y, y_numeric=True)
        if self.lam != 0:
            pass
        else:
            pass
        
        X= check_array(X)
        #y=np.asarray(y,dtype=X.dtype)
        #A=np.dot(X.T,X)
        #B=np.dot(X.T,y) #@でも行列計算行ける
        # Aw=b
        
        x_0=np.ones((len(X),1))
        x=np.hstack([x_0,X])
        A=np.dot(x.T,x)+ self.lam * np.identity(x.shape[1])
        B=np.dot(np.linalg.inv(A),x.T)
        ML=np.dot(B,y)
        
        
        self.a_ =ML[1:]
        self.b_ =ML[0]
       
        
        return self
    
    def predict(self, X):
        """
        Calc y from X
        """
        check_is_fitted(self, "a_", "b_") # 学習済みかチェックする(推奨)
        X = check_array(X)
        y=np.dot(X,self.a_)
        y+=self.b_
        
        return y
