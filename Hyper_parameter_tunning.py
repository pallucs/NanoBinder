import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,KFold,train_test_split,StratifiedKFold
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbPipeline
import optuna
import pickle
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.metrics import average_precision_score



df=pd.read_csv('../data/Dataset/dataset.csv')
df.dropna(inplace=True)

df.drop(columns=['PDB','total_score','description','packstat','sc_value'],inplace=True)

selected_features = ['complex_normalized', 'dG_cross', 'dG_cross/dSASAx100', 'dG_separated',
       'dG_separated/dSASAx100', 'dSASA_hphobic', 'dSASA_int', 'dSASA_polar',
       'delta_unsatHbonds', 'dslf_fa13', 'fa_atr', 'fa_dun', 'fa_elec',
       'fa_intra_rep', 'fa_intra_sol_xover4', 'fa_rep', 'fa_sol',
       'hbond_E_fraction', 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc',
       'hbond_sr_bb', 'hbonds_int', 'lk_ball_wtd', 'nres_all', 'nres_int',
       'omega', 'p_aa_pp', 'per_residue_energy_int', 'pro_close',
       'rama_prepro', 'ref', 'side1_normalized', 'side1_score',
       'side2_normalized', 'side2_score', 'yhh_planarity']

X_data=df.loc[:,~df.columns.isin(['label'])]
X_data=X_data[selected_features]

# Create correlation matrix
corr_matrix = X_data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find index of feature columns with correlation greater than a certain threshold
# You can vary this threshold depending on your problem
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features 
X_data.drop(X_data[to_drop], axis=1, inplace=True)

print(X_data.columns)

final_feat = ['complex_normalized', 'dG_cross', 'dG_cross/dSASAx100', 'dSASA_hphobic',
       'dSASA_int', 'dSASA_polar', 'delta_unsatHbonds', 'dslf_fa13', 'fa_atr',
       'hbond_E_fraction', 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc',
       'hbond_sr_bb', 'hbonds_int', 'nres_int', 'omega',
       'per_residue_energy_int', 'pro_close', 'rama_prepro', 'ref',
       'side1_normalized', 'side1_score', 'side2_normalized', 'side2_score',
       'yhh_planarity']

def get_score(estimator,X_test,y_test):
    pred=estimator.predict(X_test)
    return average_precision_score(y_test,pred)

X_data=X_data[final_feat]
X_data = X_data.to_numpy()

Y_data=df.loc[:,df.columns.isin(['label'])].to_numpy()

def objective(trial):
    scores=[]
    model=[]
    datas=[]
    pr_value=0
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 4,10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state':trial.suggest_int('random_state',1,1000),
        'class_weight':trial.suggest_categorical('class_weight',['balanced','balanced_subsample'])
    }
    fold=5
    folds=StratifiedKFold(n_splits=fold,random_state=43,shuffle=True)
    for x,y in folds.split(X_data,Y_data):
        rc = RandomForestClassifier()
        rc.set_params(**params)
        pipe = imbPipeline([('oversample',SMOTETomek(sampling_strategy='minority', random_state=params['random_state'])),
        ('clf',rc)])
        x_train=X_data[x]
        y_train=Y_data[x]
        pipe.fit(x_train,y_train)
        x_test=X_data[y]
        y_test=Y_data[y]
        datas.append([x,y])
        model.append(rc)
        scores.append(get_score(pipe,x_test,y_test))
    if pr_value<np.mean(scores):
        pr_value=np.mean(scores)
        with open("../results/RF_model_best.pkl", "wb") as f:
            pickle.dump(model, f)

    with open('../results/Optuna_hyperparameter.csv' ,'a') as f:
        f.write(' '.join([str(key)+':'+str(value) for (key,value) in trial.params.items()]) + ',' + str(np.mean(scores)) + '\n')

    return np.mean(scores)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10,show_progress_bar=True) # Use atleast n_trails 5000.
print("Best Hyperparameters:")
print(study.best_params)
print("***********************************")
