
import numpy as np
from collections import Counter
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2,axis=-1))
    
class KNN:

    def __init__(self, k=3):
        """
        k: int
            The number of nearest neighbors to consider.
        """
        self.k = k

    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: np.ndarray
            The training data.
        y: np.ndarray
            The training labels.
        """
        
        self.X_train = X
        self.y_train = y

  
    def _max_occurrences(self, arr: np.ndarray):
        """
        arr: np.ndarray
            A 1D array of labels.
        returns: int
            The most common label in the array.
        """
        counts = np.bincount(arr)
        return np.argmax(counts)
    
    def predict(self, X: np.ndarray):
        """ 
        X: np.ndarray
            The test data.
        returns: np.ndarray
            The predicted labels.
        """
        distances = euclidean_distance(X[:,np.newaxis,:],self.X_train[np.newaxis,:,:])
        k_indices = np.argsort(distances ,axis=-1)[:,:self.k]
        k_nearest_labels = self.y_train[k_indices]
        predicted_labels = np.apply_along_axis(self._max_occurrences,axis=1,arr=k_nearest_labels)
        return np.array(predicted_labels)
    
    

    def _predict(self, x):
        # compute distances
        distances = euclidean_distance(x, self.X_train)

        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        #majority vote  , most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    

if __name__ == "__main__":
    from mlkit.datasets.iris import IRIS
    import mlkit
    iris_data = IRIS.create_from_sklearn()
    X_train, X_test, y_train, y_test  = iris_data.get_train_test_split()
    classifier = KNN(k=3)
    classifier.fit(X_train,y_train)
    predictions = classifier.predict(X_test)

    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)