import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association
from scipy.stats.contingency import margins


def equal_confusion_test(contingency):
    test = dict()
    test_result = chi2_contingency(contingency, correction = False)
    test['chi2']       = test_result[0].item()
    test['p']          = test_result[1].item()
    test['dof']        = test_result[2]
    test['expected']   = pd.DataFrame(test_result[3], index = contingency.index, columns = contingency.columns)
    return test


def confusion_parity_error(contingency):
    return association(contingency, method = 'cramer')

def posthoc_analysis(contingency, test):
    n = contingency.sum().sum()
    rsum, csum = margins(contingency)
    rsum = rsum.astype(np.float64)
    csum = csum.astype(np.float64)
    v = csum * rsum * (n - rsum) * (n - csum) / n**3
    residuals_stat = (contingency - test['expected']) / np.sqrt(v)
    return pd.DataFrame(residuals_stat, index = contingency.index, columns = contingency.columns)

def ratio_to_actual(contingency, labels):
    df = contingency
    df.columns = pd.MultiIndex.from_product([labels, labels], names=['Actual', 'Predicted'])
    return (df/df.groupby(level=0, axis=1).sum())

def ratio_to_predicted(contingency, labels):
    df = contingency
    df.columns = pd.MultiIndex.from_product([labels, labels], names=['Actual', 'Predicted'])
    df = df.reorder_levels(['Predicted','Actual'], axis = 1)
    return (df/df.groupby(level=0, axis=1).sum())

def ratio_to_all(contingency):
    df = (contingency.transpose()/contingency.sum(axis=1)).transpose()
    df2 = df.reorder_levels(['Predicted','Actual'], axis = 1)
    return df, df2
    

# Main function
def ecf(sampledata, s, pred, gt):
    labels = list(set(sampledata[gt]) | set(sampledata[pred]))
    groups = list(set(sampledata[s]))
    
    index      = dict()
    confusion  = dict()
    for group in groups:
        index[group] = np.where(sampledata[s] == group)
        confusion[group] = confusion_matrix(sampledata.loc[index[group], gt], sampledata.loc[index[group], pred], labels = labels)

    contingency = list()
    for group in groups:
        contingency.append(confusion[group].ravel())
    contingency = np.vstack(contingency)
    
    
    clabels = ['Actual ' + i + ' - Predicted ' +j for i in labels for j in labels]
    contingency = pd.DataFrame(contingency, index = groups, columns = pd.MultiIndex.from_product([labels, labels], names=['Actual', 'Predicted']))
    test = equal_confusion_test(contingency)
    metric = confusion_parity_error(contingency)
    posthoc = posthoc_analysis(contingency, test)
    ratio_gt, ratio_pred, ratio_all = ratio_to_actual(contingency, labels), ratio_to_predicted(contingency, labels), ratio_to_all(contingency)
    
    return {'labels':labels, 'groups':groups, 'contingency':contingency, 'equal_confusion_test':test, 'confusion_parity_error':metric, 'posthoc_analysis':posthoc, 'ratio_gt':ratio_gt, 'ratio_pred':ratio_pred,  'ratio_all':ratio_all}