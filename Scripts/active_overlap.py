from time import time
from glob import glob
from os.path import splitext, basename
import pandas as pd
import matplotlib.pyplot as plt
from venn import venn
from os import makedirs

def main():
    start_time = time()
    files = glob('Data/*.parquet')
    actives_3 = {}
    actives_5 = {}
    for file in files:
        target = splitext(basename(file))[0]
        data = fileGrab(file)
        data_actives_3 = activeGrab(data, 3)
        if len(data_actives_3) >= 1:
            actives_3[target] = set(data_actives_3['SMILES'].tolist())
        else:
            pass
        data_actives_5 = activeGrab(data, 5)
        if len(data_actives_5) >= 1:
            actives_5[target] = set(data_actives_3['SMILES'].tolist())
        else:
            pass
    vennPlot(actives_3, 3)
    vennPlot(actives_5, 5)
    return print('Total Runtime (seconds): ', round(time() - start_time, 2))
def fileGrab(file):
    data_raw = pd.read_parquet(file)
    data_sorted = data_raw.sort_values(by = 'target_count', ascending = False)
    data_validSMILES = data_sorted[data_sorted['SMILES'].notna()]
    data_unique = data_validSMILES.drop_duplicates(subset = 'SMILES')
    return data_unique
def activeGrab(data, count_value):
    data['active'] = [1 if data['target_zscore'].values[index] >= 1 and
                      data['target_count'].values[index] >= count_value else 0 for
                      index in range(len(data))]
    return data[data['active'] == 1]
def vennPlot(actives, count_value):
    venn(actives)
    plt.title(f'Overlap of Selection Actives (Count >= {count_value})')
    makedirs('Figures/activeOverview', exist_ok = True)
    plt.savefig(f'Figures/activeOverview/active_overlap_{count_value}.png')
    plt.clf()
    return
main()