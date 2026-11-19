import numpy as np

def knn_classifier(X_train: list, y_train: list, X_test: list, k: int) -> list:
    """
    Returns one predicted label for each test row.
    """
    X_train = np.asarray(X_train, dtype=np.float64) # (n,d)
    y_train = np.asarray(y_train, dtype=np.float64) # (n)
    X_test = np.asarray(X_test, dtype=np.float64) # (m, d)

    # (m, n)
    dist = np.sqrt(np.sum((X_test[:,None,:]-X_train[None,:,:])**2, axis=-1))

    # for each m, need to sort n
    sorted = np.argsort(dist, axis=-1, kind="stable")[:,:k] # (m,k)
    labels = np.take_along_axis(y_train[None,:], sorted, axis = -1) # (m,k)
    unique_labels = np.unique(y_train) # (c)
    labels = labels[:,:,None] == unique_labels[None,None,:] # (m, k , c)
    # sum over k
    labels = np.sum(labels, axis=1) # (m,c)
    return unique_labels[np.argmax(labels, axis=-1)].tolist()
    
    
    
    


    
    
    
    
    
    
    

    

    
        
