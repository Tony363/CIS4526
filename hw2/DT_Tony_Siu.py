import os
import pandas as pd
import numpy as np
#https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb

from sklearn.model_selection import train_test_split
from multiprocessing import Process, Queue,Lock
from eval import confusion_matrix,classification_report,accuracy_score


class Node:
    def __init__(
        self, 
        feature:np.ndarray=None, 
        threshold:np.float32=None, 
        left:object=None, 
        right:object=None, 
        *, 
        value:np.float32=None
    )->None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self)->bool:
        return self.value is not None
    

class DecisionTreeModel:
    def __init__(
        self, 
        seed:int,
        max_depth:int=100, 
        criterion:str='gini', 
        min_samples_split:int=2, 
        impurity_stopping_threshold:int=0.01
    )->None:
        assert impurity_stopping_threshold > 0, 'impurity_stopping_threshold must be greater than 0 so that the DT won\'t overfit'
        
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None
        
        # Additional global variables
        self.rng = np.random.RandomState(seed)
        self.loss = self._entropy if self.criterion == 'entropy' else self._gini
        self.n_samples = self.n_features = self.classes = None
        

    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    )->None:
        """_summary_ 
        converts X features and y target
        variables to numpy arrays if 
        they are pandas objects. Calls
        >>> self._fit(X,y)
        after dealt with different 
        object types
        
        Parameters
        ----------
        X: input features
        y: target vector
        
        Returns:
            None: ()
        """
        # TODO
        # call the _fit method
        self.classes = np.unique(y)
        y_type = isinstance(y,pd.core.series.Series)
        if y_type and y.dtype == 'object':
            y = y.astype('category').cat.codes
        if y_type and isinstance(X,pd.DataFrame):
            X,y = X.to_numpy(),y.to_numpy()
            
        self._fit(X,y)
        # end TODO        


    def predict(
        self, 
        X: pd.DataFrame
    )->np.ndarray:
        """_summary_ 
        converts X features 
        variables to numpy arrays if 
        they are pandas objects. returns
        >>> self._predict(X)
        after dealt with different 
        object types
        
        Parameters
        ----------
        X: input features
        y: target vector
        
        Returns:
            np.ndarray: Decision Tree predictions 
        """
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        # end TODO
        return self._predict(X)
        
        
    def _fit(self, X, y):
        """_summary_ 
        Starts building tree pointing to 
        >>> self.root 
        minipointer
        
        Parameters
        ----------
        X: input features
        y: target vector
        
        Returns:
            None: ()
        """
        self.root = self._build_tree(X, y)
        
        
    def _predict(
        self, 
        X:np.ndarray
    )->np.ndarray:
        """_summary_ 
       
        Parameters
        ----------
        X: input features
        
        Returns:
            np.ndarray: predictions
        """
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)    
        
        
    def _is_finished(
        self, 
        y:np.ndarray,
        depth:int,
    )->bool:
        """_summary_ 
        check to whether further splits is necessary
        based on max_dept hyper parameter, min samples 
        split hyper parameter and whether loss is less than
        impurity threshold
        
        Parameters
        ----------
        y: target vector
        
        Returns:
            bool: whether further split is necessary
        """

        return (depth >= self.max_depth
            or len(self.classes) == 1
            or self.n_samples < self.min_samples_split
            or (self.impurity_stopping_threshold > 0 and self._is_homogenous_enough(y)))
        
    
    def _is_homogenous_enough(
        self,
        y:np.ndarray
    )->bool:
        """_summary_ 
        
        Parameters
        ----------
        y: target vector
        
        Returns:
            bool: whether loss less than impurity threhold
        """
        return self.loss(y)  < self.impurity_stopping_threshold
                     
                              
    def _build_tree(
        self, 
        X:np.ndarray, 
        y:np.ndarray, 
        depth:int=0
    )->Node:
        """_summary_ 
        recursively builds tree based on stopping criterias
        and best splits from thresholds of features
        
        Parameters
        ----------
        X: input features
        y: target vector
        
        Returns:
            Node: child node of current node
        """
        self.n_samples, self.n_features = X.shape
        
        # stopping criteria
        if self._is_finished(y,depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = self.rng.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)
    
    
    @staticmethod
    def _gini(
        y:np.ndarray
    )->np.float32:
        """_summary_ 
        aggregates gini coefficient loss
        
        Parameters
        ----------
        y: target vector
        
        Returns:
            float: gini coefficient loss
        """        
        proportions = np.bincount(y) / len(y)
        gini = np.sum([p * (1 - p) for p in proportions if p > 0])
        #end TODO
        return gini


    @staticmethod
    def _entropy(
        y:np.ndarray
    )->np.float32:
        """_summary_ 
        aggregates entropy  loss
        
        Parameters
        ----------
        y: target vector
        
        Returns:
            float: entropy loss
        """    
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        # end TODO
        return entropy
    
    
    @staticmethod
    def _create_split(
        X:np.ndarray, 
        thresh:np.float32
    )->np.float32:
        """_summary_ 
        returns best slit indexes
        
        Parameters
        ----------
        X: feature vector
        thresh: best split threshold
        
        Returns:
            tuple: (left index split,right index split)
        """    
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx


    def _information_gain(self, X, y, thresh):
        """_summary_ 
        aggregates information gain from parent and child node loss
        
        Parameters
        ----------
        X: feature vector
        y: target vector
        
        Returns:
            float: information gain
        """            
        parent_loss = self.loss(y) 
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0: 
            return 0
        
        child_loss = (n_left / n) * self.loss(y[left_idx]) + (n_right / n) * self.loss(y[right_idx])
        # end TODO
        return parent_loss - child_loss
       
       
    def _best_split(
        self, 
        X:np.ndarray, 
        y:np.ndarray, 
        features:list
    )->tuple:
        """_summary_ 
        Loops through every threshold to find 
        the best split threshold on condition
        of the information gain
        
        Parameters
        ----------
        X: input features
        y: target vector
        features: feature indexes
        
        Returns:
            tuple: (split features, split threshold)
        """
        split = {'score':- 1, 'feat': None, 'thresh': None}
        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']
    
    
    def _traverse_tree(
        self, 
        x:np.ndarray, 
        node:Node,
    )->np.int16:
        """_summary_ 
        Makes predictions on Decision tree stored Node 
        thresholds via recursive traversal to leaf 
        nodes.
        
        returns the predicted value if node is a leaf node
        
        Parameters
        ----------
        X: input features vector
        node: traversed leaf node                
        Returns
        -------
            int: leaf feature value
        """
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestModel(object):
    def __init__(
        self, 
        seed:int,
        n_estimators:int,
        impurity_stopping_threshold:int=0.01,
    )->None:
        # TODO:
        self.rng = np.random.RandomState(seed)
        seeds = self.rng.randint(1,100000,size=n_estimators)
        
        self.trees = np.asarray([
            DecisionTreeModel(seed=seed,impurity_stopping_threshold=impurity_stopping_threshold) 
            for seed in seeds
        ])
        
        self.__shuffle_idxs = None
        # end TODO
        

    def fit(self, X: pd.DataFrame, y: pd.Series)->None:  
        """_summary_ 
        converts X features and y target
        variables to numpy arrays if 
        they are pandas objects. For n_estimators, 
        Calls shuffled X features and y target variables
        >>> self._fit(X,y)
        
        Parameters
        ----------
        X: input features
        y: target vector
        
        Returns:
            None: ()
        """
        self.__shuffle_idxs = np.arange(X.shape[0])
        # shuffle samples bagging 
        if isinstance(X,pd.DataFrame) and isinstance(y,pd.core.series.Series):
            X,y = X.to_numpy(),y.to_numpy()
        
        for tree in self.trees:
            self.rng.shuffle(self.__shuffle_idxs)
            tree.fit(X[self.__shuffle_idxs],y[self.__shuffle_idxs])


    def predict(self, X: pd.DataFrame)->np.ndarray:
        """_summary_ 
        converts X features 
        variables to numpy arrays if 
        they are pandas objects. Does
        voting based ensemble of n_estimator
        trees
        
        Parameters
        ----------
        X: input features
        y: target vector
        
        Returns:
            np.ndarray: Decision Tree predictions 
        """       
        preds = np.zeros((len(self.trees),X.shape[0]))
        for idx,tree in enumerate(self.trees):
            preds[idx] = tree.predict(X)
            
        return np.apply_along_axis(
            func1d=lambda x:np.bincount(x).argmax(),
            axis=0,
            arr=preds.astype('int64')
        )
        # end TODO    



def _test():
    seed = 12345
    df = pd.read_csv('breast_cancer.csv')
    
    #X = df.drop(['diagnosis'], axis=1).to_numpy()
    #y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1).to_numpy()

    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    clf = DecisionTreeModel(seed=seed,max_depth=10,)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    rfc = RandomForestModel(seed=seed,n_estimators=16,)
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    print(classification_report(y_test, rfc_pred))

    
if __name__ == "__main__":
    _test()
