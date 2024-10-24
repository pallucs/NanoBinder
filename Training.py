import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,KFold,train_test_split,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, PrecisionRecallDisplay, matthews_corrcoef, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn import metrics

df=pd.read_csv('../data/Dataset/dataset.csv')
df.dropna(inplace=True)
df.drop(columns=['PDB','total_score','description','packstat','sc_value'],inplace=True)
df = df.reset_index(drop=True)


naonobody_features_rf= ['complex_normalized', 'dG_cross', 'dG_cross/dSASAx100', 'dSASA_hphobic',
       'dSASA_int', 'dSASA_polar', 'delta_unsatHbonds', 'dslf_fa13', 'fa_atr',
       'hbond_E_fraction', 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc',
       'hbond_sr_bb', 'hbonds_int', 'nres_int', 'omega',
       'per_residue_energy_int', 'pro_close', 'rama_prepro', 'ref',
       'side1_normalized', 'side1_score', 'side2_normalized', 'side2_score',
       'yhh_planarity']

def get_score(estimator,X_test,y_test):
    pred=estimator.predict(X_test)
    return average_precision_score(y_test,pred)

def get_accuracy(estimator, X_test, y_test):
    pred=estimator.predict(X_test)
    return accuracy_score(y_test,pred)

def get_mcc(estimator, X_test, y_test):
    pred=estimator.predict(X_test)
    return matthews_corrcoef(y_test,pred)

def get_f1(estimator, X_test, y_test):
    pred=estimator.predict(X_test)
    return f1_score(y_test,pred)

def get_cm(estimator, X_test, y_test):
    pred=estimator.predict(X_test)
    return confusion_matrix(y_test,pred)

X_data=df.loc[:,~df.columns.isin(['label'])]
X_data=X_data[naonobody_features_rf]

X_data = X_data.to_numpy()
Y_data=df.loc[:,df.columns.isin(['label'])].to_numpy()

# Define parameters for the RandomForestClassifier
params = {
    'n_estimators': 185,
    'criterion': 'gini',
    'max_depth': 10,
    'min_samples_split': 3,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': False,
    'random_state': 474,
    'class_weight': 'balanced'
}

# Set global font size for plots
plt.rcParams.update({'font.size': 18})

# Initialize RandomForestClassifier with specified parameters
rc = RandomForestClassifier(**params)

# Number of cross-validation folds
fold = 5
folds = StratifiedKFold(n_splits=fold, random_state=43, shuffle=True)

# Initialize plots for ROC and Precision-Recall Curves
roc_fig = plt.figure(figsize=(10, 8))  # Create a new figure for ROC curve
roc_ax = plt.gca()  # ROC curve axis

prc_fig = plt.figure(figsize=(10, 8))  # Create a new figure for Precision-Recall curve
prc_ax = plt.gca()  # Precision-Recall curve axis

# Cross-validation and plotting
for i, (train_idx, test_idx) in enumerate(folds.split(X_data, Y_data)):
    # Create a pipeline with SMOTETomek for oversampling and the classifier
    pipe = imbPipeline([
        ('oversample', SMOTETomek(sampling_strategy='minority', random_state=43)),
        ('clf', rc)
    ])
    
    # Split the data into training and test sets
    x_train, y_train = X_data[train_idx], Y_data[train_idx]
    x_test, y_test = X_data[test_idx], Y_data[test_idx]
    
    # Fit the model
    pipe.fit(x_train, y_train)
    
    # Calculate and plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, pipe.predict_proba(x_test)[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f'Fold {i+1}').plot(ax=roc_ax)
    
    # Calculate and plot Precision-Recall curve
    prec, recall, _ = precision_recall_curve(y_test, pipe.predict_proba(x_test)[:, 1])
    prc_auc = metrics.average_precision_score(y_test, pipe.predict_proba(x_test)[:, 1])
    PrecisionRecallDisplay(precision=prec, recall=recall, average_precision=prc_auc, estimator_name=f'Fold {i+1}').plot(ax=prc_ax)

    accuracy= get_accuracy(pipe,x_test, y_test)
    MCC= get_mcc(pipe,x_test, y_test)
    F1= get_f1(pipe,x_test, y_test) 
    PRC=get_score(pipe,x_test, y_test) 
    print("*****************************************************************************************\n")
    print(f"Fold {i+1}:\n\t\t PRC: {PRC} \n\t\t MCC: {MCC} \n\t\t F1: {F1} \n\t\t ACC: {accuracy}\n")

    

# Finalize and save the ROC plot
plt.legend()
roc_fig.savefig('../results/roc-plot.png', bbox_inches='tight', dpi=300)
plt.close(roc_fig) 

# Finalize and save the Precision-Recall plot
plt.legend()
prc_fig.savefig('../results/prc-plot.png', bbox_inches='tight', dpi=300)
plt.close(prc_fig) 

