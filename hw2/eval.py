

import pandas as pd
import numpy as np

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def classification_report(y_test, y_pred):
    # calculate precision, recall, f1-score
    # TODO:
    series = isinstance(y_test,pd.core.series.Series) and isinstance(y_pred,pd.core.series.Series)
    if series and y_test.dtype == y_pred.dtype == 'object':
        y_test,y_pred = y_test.astype('category').cat.codes, y_pred.astype('category').cat.codes
    if series:
        y_test,y_pred = y_test.to_numpy(),y_pred.to_numpy()
    
    n = np.unique(y_test).size
    cm = confusion_matrix(y_test,y_pred,n=n)
    acc = accuracy_score(y_test,y_pred)
    
    tp_total = fp_total = fn_total = 0
    precision,recall,f1_score = np.zeros(n),np.zeros(n),np.zeros(n)
    for i in range(n):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        tp_total += tp
        fp_total += fp
        fn_total += fn
    
    report = "".join(tuple(
        "Class {}\t\t{:.3f}\t{:.3f}\t{:.3f}\t{}\n".format(i, precision[i], recall[i], f1_score[i],np.sum(y_test == i))
        for i in range(n)
    ))
    m_precision,m_recall,m_f1 = precision.sum()/precision.size,recall.sum()/recall.size,f1_score.sum()/f1_score.size
    w_precision,w_recall = tp_total/(tp_total + fp_total),tp_total/(tp_total + fn_total)
    w_f1 = 2*w_precision*w_recall/(w_precision + w_recall)
    # end TODO
    return '\t\tprec\trecall\tf1\tsupport\n' + report +\
        '\nacc\t\t\t\t{:.3f}\t{}\n'.format(acc,y_test.size)+\
            'macro avg\t{:.3f}\t{:.3f}\t{:.3f}\t{}\n'.format(m_precision,m_recall,m_f1,y_test.size) +\
                'weight avg\t{:.3f}\t{:.3f}\t{:.3f}\t{}'.format(w_precision,w_recall,w_f1,y_test.size)
    

def confusion_matrix(
    y_test:np.ndarray, 
    y_pred:np.ndarray,
    n:int=2
)->np.ndarray:
    # return the 2x2 matrix
    # TODO: Multiclass for ints or strings
    # https://stackoverflow.com/questions/68157408/using-numpy-to-test-for-false-positives-and-false-negatives
    result = np.zeros((n, n), dtype=np.uint8)
    if n < 3:
        result[1,1] = np.sum(np.logical_and(y_pred == 1, y_test == 1))
        result[0,0] = np.sum(np.logical_and(y_pred == 0, y_test == 0))
        result[0,1] = np.sum(np.logical_and(y_pred == 1, y_test == 0))
        result[1,0] = np.sum(np.logical_and(y_pred == 0, y_test == 1))
        # end TODO
        return result  
    
    for i in range(n):
        for j in range(n):
            result[i, j] = np.sum((y_test == i) & (y_pred == j))
    return result

