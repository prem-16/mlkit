import numpy as numpy






class IRIS:
    def __init__(self,X,y):
        self.X = X
        self.y = y

    @classmethod
    def create_from_sklearn(cls):
        from sklearn import datasets
        iris = datasets.load_iris()
        
        return cls(iris.data,iris.target)

    def get_train_test_split(self, test_size=0.2, random_state=1234):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test 

    
if __name__ == "__main__":
    import matplotlib.pyplot  as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    iris_data = IRIS.create_from_sklearn()
    X_train, X_test, y_train, y_test  = iris_data.get_train_test_split()
    print(X_train.shape)
    print(X_train[0])

    print(y_train.shape)
    print(y_train[0])
    plt.figure()
    plt.scatter(iris_data.X[:,0], iris_data.X[:,1], c=iris_data.y ,cmap=cmap ,edgecolor='k',s=20)
    plt.show()