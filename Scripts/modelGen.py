from time import time
from glob import glob
import pandas as pd
from os.path import splitext, basename
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from joblib import dump
from os import makedirs
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    start_time = time()
    files = glob('Data/*.parquet')
    for file in files:
        target = splitext(basename(file))[0]
        data = dataGrab(file, 3)
        data_fp = fpGen(data)
        if len(data[data['active'] == 1]) > 5:
            best_model, X_test, Y_test, Y_pred = modelGen_RandomForest(data_fp, target)
            plot_model(best_model, X_test, Y_test, Y_pred, target)
        else:
            print('Not enough actives to model for:', target)
            continue
    return print('Total Runtime (seconds): ', round(time() - start_time, 2))
def dataGrab(file, value_count):
    data_raw = pd.read_parquet(file)
    data_raw['active'] = [1 if data_raw['target_zscore'].values[index] >= 1 and
                      data_raw['target_count'].values[index] >= value_count else 0 for
                      index in range(len(data_raw))]
    data_sorted = data_raw.sort_values(by = 'target_count', ascending = False)
    data_validSMILES = data_sorted[data_sorted['SMILES'].notna()]
    data_unique = data_validSMILES.drop_duplicates(subset = 'SMILES')
    data_inactivesAll = data_unique[data_unique['active'] == 0]
    data_actives = data_unique[data_unique['active'] == 1]
    data_inactives = data_inactivesAll.sample(n = 10*len(data_actives), random_state = 42)
    data = pd.concat([data_actives, data_inactives], ignore_index = True)
    return data
def fpGen(data):
    mols = data['SMILES'].apply(MolFromSmiles)
    morganGen = GetMorganGenerator(radius = 2, fpSize = 2048)
    data_fp = pd.DataFrame([list(fp) for fp in morganGen.GetFingerprints(mols)])
    data_fp['activity'] = data['active']
    return data_fp
def ppv_top_n(y_true, y_pred, top_n: int = 128, threshold: float = 0.5) -> float:
    """
    Function taken from https://github.com/molecularmodelinglab/plate-ppv/blob/main/PlatePPV.py 
    calculate the ppv (precision) taking into account only the top N prediction with the highest score/probability
    :param y_true: real classes
    :param y_pred: predicted probability of positive class
    :param top_n: number of top prediction to choose (default = 128)
    :param threshold: the probability threshold to consider a compound active or not
    :return: float; the metric value (0-1)
    """
    y_pred = np.atleast_1d(y_pred)
    y_true = np.atleast_1d(y_true)
    _tmp = np.vstack((y_true, y_pred)).T[y_pred.argsort()[::-1]][:top_n, :]
    _tmp = _tmp[np.where(_tmp[:, 1] > threshold)[0]].copy()
    return np.sum(_tmp[:, 0]) / len(_tmp)
def modelGen_RandomForest(data_fp, target):
    X = data_fp.iloc[:, :-1]
    Y = data_fp['activity']
    best_score = 0
    best_model = None
    fold_number = 1
    validation = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    for train_index, _ in validation.split(X, Y):
        X_fold, Y_fold = X.iloc[train_index], Y.iloc[train_index]
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_fold, Y_fold, test_size = 0.2, stratify = Y_fold, shuffle = True,
            random_state = 42
        )
        model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(random_state = 42)
        )
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        score = ppv_top_n(Y_test, Y_pred)
        if score > best_score:
            best_score = score
            best_model = model
        fold_number += 1
    makedirs('Data/models/', exist_ok = True)
    dump(best_model, f'Data/models/best_randomForestClass_{target}.joblib')
    return best_model, X_test, Y_test, Y_pred
def plot_model(best_model, X_test, Y_test, Y_pred, target):
    makedirs('Figures/models/', exist_ok = True)
    Y_prob = best_model.predict_proba(X_test)[:, 1]
    false_pos_rate, true_pos_rate, _ = roc_curve(Y_test, Y_prob)
    roc_auc = auc(false_pos_rate, true_pos_rate)
    ppv = ppv_top_n(Y_test, Y_pred)
    plt.plot(
        false_pos_rate, true_pos_rate, label = f'AUC: {roc_auc:.2f}\nPPV: {ppv:.2f}'
    )
    plt.title(f'{target} ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(f'Figures/Models/{target}_ROC.png')
    plt.clf()
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    sns.heatmap(
        conf_matrix, cmap = 'Blues', 
        xticklabels = ['Predicted Inactives', 'Predicted Actives'],
        yticklabels = ['True Inactives', 'True Actives'],
        annot = True
    )
    plt.title(f'{target} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'Figures/Models/{target}_ConfMatrix.png')
    plt.clf()
    figs, axes = plt.subplots(1, 2, figsize = (12, 6))
    axes[0].plot(
        false_pos_rate, true_pos_rate, label = f'AUC: {roc_auc:.2f}\nPPV: {ppv:.2f}'
    )
    axes[0].plot([0,1], [0,1], linestyle = '--')
    axes[0].set_title('ROC Curve')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    sns.heatmap(
        conf_matrix, cmap = 'Blues',
        xticklabels = ['Predicted Inactive', 'Predicted Active'],
        yticklabels = ['True Inactive', 'True Active'],
        ax = axes[1], annot = True
    )    
    axes[1].set_title('Confusion Matrix')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    plt.suptitle(f'{target} Model Metrics')
    plt.savefig(f'Figures/Models/{target}_combinedPlots.png')
    plt.clf()
    return
main()