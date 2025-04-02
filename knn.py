import numpy as np

#Eucladian distance function
def euclidean_distance(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return np.sqrt(np.sum((x1 - x2) ** 2))


#Manhattan distance function
def manhattan_distance(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return np.sum(np.abs(x1 - x2))


from collections import Counter

def knn_euc(X_train, y_train, X_test, k):
    predictions = []
    X_train = X_train.values  #Convert to NumPy arrays
    X_test = X_test.values    #Convert to NumPy arrays

    for test_point in X_test:
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        k_indices = np.argsort(distances)[:k]  #Sort the k-nearest distances
        k_nearest_classes = [y_train.iloc[i] for i in k_indices]  #Get the class of k nearest
        most_common = Counter(k_nearest_classes).most_common(1)
        predictions.append(most_common[0][0])  #Append the most common class
    
    return predictions

def knn_man(X_train, y_train, X_test, k):
    predictions = []
    X_train = X_train.values  #Convert to NumPy arrays
    X_test = X_test.values    #Convert to NumPy arrays

    for test_point in X_test:
        distances = [manhattan_distance(test_point, train_point) for train_point in X_train]
        k_indices = np.argsort(distances)[:k]  #Sort the k-nearest distances
        k_nearest_labels = [y_train.iloc[i] for i in k_indices]  #Get the class of k nearest
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])  #Append the most common class
    
    return predictions
