

import pandas as pd
import numpy as np
import seaborn as sns
#https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb

from sklearn import datasets
from sklearn.model_selection import train_test_split
from typing import Tuple,Iterable



class Node:
    def __init__(
        self, 
        feature=None, 
        threshold=None, 
        left=None, 
        right=None, 
        *, 
        value=None
    )->None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
    

class DecisionTreeModel:
    """
    everything should work for strings and int vectors
    """
    def __init__(
        self, 
        max_depth:int=100, 
        criterion:str='gini', 
        min_samples_split:int=2, 
        impurity_stopping_threshold:int=0
    )->None:
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None
        
        # Additional global variables
        self.n_samples = self.n_features = self.classes = self.split = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO
        # call the _fit method
        if isinstance(y[0],str):
            self.classes = {cl:idx for idx,cl in enumerate(np.unique(y))}
            y = np.array([self.classes[cl] for cl in y])
        self._fit(X,y)
        # end TODO
        print("Done fitting")
        

    def predict(self, X: pd.DataFrame):
        # TODO
        # call the _predict method
        # end TODO
        return self._predict(X)
        
        
    def _fit(self, X, y):
        self.root = self._build_tree(X, y)
        
    def _predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)    
        
    def _is_finished(self, depth): # entropy case? gini case? which impurity?
        # TODO: add another stopping criteria
        # modify the signature of the method if needed
        return (depth >= self.max_depth
            or len(self.classes) == 1
            or self.n_samples < self.min_samples_split
            or self._is_homogenous_enough())
        
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        # end TODO
        return False
    
    def _is_homogenous_enough(self):
        # TODO: 
        # end TODO
        return self.split['score'] < self.impurity_stopping_threshold
                              
    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        # self.n_class_labels = len(np.unique(y))
        
        # stopping criteria
        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)
    

    def _gini(self, y):
        #TODO convert to categorical as well
        proportions = np.bincount(y) / len(y)
        gini = np.sum([p * (1 - p) for p in proportions if p > 0])
        #end TODO
        return gini

    def _entropy(self, y:Iterable):
        # TODO: the following won't work if y is not integer
        # make it work for the cases where y is a categorical variable
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        # end TODO
        return entropy
        
    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        # TODO: fix the code so it can switch between the two criterion: gini and entropy 
        loss = self._entropy if self.criterion == 'entropy' else self._gini
        parent_loss = loss(y) 
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0: 
            return 0
        
        child_loss = (n_left / n) * loss(y[left_idx]) + (n_right / n) * loss(y[right_idx])
        # end TODO
        return parent_loss - child_loss
       
    def _best_split(self, X, y, features):
        '''TODO: add comments here

        '''
        self.split = {'score':- 1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > self.split['score']:
                    self.split['score'] = score
                    self.split['feat'] = feat
                    self.split['thresh'] = thresh

        return self.split['feat'], self.split['thresh']
    
    def _traverse_tree(self, x, node):
        '''TODO: add some comments here
        '''
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestModel(object):
    # call the above DecisionTreeModel class, not the sklearn classes
    def __init__(
        self, 
        n_estimators:int
    )->None:
        # TODO:
        self.trees = (DecisionTreeModel(),) * n_estimators
        # end TODO

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO:
        for tree in self.trees:
            tree.fit(X,y)
        # end TODO
        print("Fitted RF")


    def predict(self, X: pd.DataFrame):
        # TODO:
        preds = np.array(list(map(lambda tree:tree.predict(X),self.trees)))
        return np.apply_along_axis(func=np.bincount,axis=1,arr=preds)
        # end TODO

    

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def classification_report(y_test, y_pred):
    # calculate precision, recall, f1-score
    # TODO:
    cm = confusion_matrix(y_test,y_pred)
    precision = cm[1,1]/(cm[1,1] + cm[0,1])
    recall = cm[1,1]/(cm[1,1] + cm[1,0])
    f1 = 2*(precision * recall)/(precision + recall)
    acc = accuracy_score(y_test,y_pred)
    # end TODO
    '''
                  precision    recall  f1-score   support

     class 0       0.50      1.00      0.67         1
     class 1       0.00      0.00      0.00         1
     class 2       1.00      0.67      0.80         3

    accuracy                           0.60         5
   macro avg       0.50      0.56      0.49         5
weighted avg       0.70      0.60      0.61         5
    '''
    return f'\t\tprec\trecall\f1-score\tsupport\n' + f'accuracy\t\t\t\t{acc}\n'+ 'macro avg\t'

def confusion_matrix(y_test, y_pred):
    # return the 2x2 matrix
    # TODO:
    # https://stackoverflow.com/questions/68157408/using-numpy-to-test-for-false-positives-and-false-negatives
    result = np.array([[0, 0], [0, 0]])
    result[1,1] = np.sum(np.logical_and(y_pred == 1, y_test == 1))
    result[0,0] = np.sum(np.logical_and(y_pred == 0, y_test == 0))
    result[0,1] = np.sum(np.logical_and(y_pred == 1, y_test == 0))
    result[1,0] = np.sum(np.logical_and(y_pred == 0, y_test == 1))
    # end TODO
    return result


def _test():
    
    df = pd.read_csv('breast_cancer.csv')
    
    #X = df.drop(['diagnosis'], axis=1).to_numpy()
    #y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1).to_numpy()

    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTreeModel(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

if __name__ == "__main__":
    _test()
