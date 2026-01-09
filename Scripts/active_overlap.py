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
    actives = {}
    for file in files:
        target = splitext(basename(file))[0]
        data = fileGrab(file)
        data_actives = activeGrab(data)
        if len(data_actives) >= 1:
            actives[target] = set(data_actives['SMILES'].tolist())
        else:
            pass
    vennPlot(actives)
    return print('Total Runtime (seconds): ', round(time() - start_time, 2))
def fileGrab(file):
    data_raw = pd.read_parquet(file)
    data_sorted = data_raw.sort_values(by = 'target_count')
    data_validSMILES = data_sorted[data_sorted['SMILES'].notna()]
    data_unique = data_validSMILES.drop_duplicates(subset = 'SMILES')
    return data_unique
def activeGrab(data):
    data['active'] = [1 if data['target_zscore'].values[index] >= 3 and
                      data['target_count'].values[index] >= 1 else 0 for
                      index in range(len(data))]
    return data[data['active'] == 1]
def vennPlot(actives):
    venn(actives)
    plt.title('Overlap of Selection Actives')
    makedirs('Figures/Venn', exist_ok = True)
    plt.savefig('Figures/Venn/active_overlap.png')
    plt.clf()
    return
main()