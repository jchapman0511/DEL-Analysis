from time import time
from glob import glob
import pandas as pd
from os.path import splitext, basename
import matplotlib.pyplot as plt
from os import makedirs

def main():
    start_time = time()
    data_3 = pd.DataFrame()
    data_5 = pd.DataFrame()
    files = glob('Data/*.parquet')
    for file in files:
        data = dataGrab(file)
        data_actives_3 = activeGrab(data, 3)
        data_actives_5 = activeGrab(data, 5)
        data_processed_3 = activeProcess(data_actives_3)
        data_processed_5 = activeProcess(data_actives_5)
        data_3 = pd.concat([data_3, data_processed_3])
        data_5 = pd.concat([data_5, data_processed_5])
    plot_table(data_3, 3)
    plot_table(data_5, 5)
    return print('Total Runtime (seconds): ', round(time() - start_time, 2))
def dataGrab(file):
    data_raw = pd.read_parquet(file)
    data_raw['target'] = splitext(basename(file))[0]
    data_validSMILES = data_raw[data_raw['SMILES'].notna()]
    data = data_validSMILES.drop_duplicates(subset = 'SMILES')
    return data
def activeGrab(data, value_count):
    data[['Lib', 'BB1', 'BB2', 'BB3']] = data['compound'].str.split('-', expand = True)
    data['active'] = [1 if data['target_zscore'].values[index] >= 1 and
                      data['target_count'].values[index] >= value_count else 0 for
                      index in range(len(data))]
    return data[data['active'] == 1]
def activeProcess(data):
    data_processed = pd.DataFrame({
        'Target': data['target'].unique(), 'Active Count': len(data),
        'Unique Active Libraries': len(data['Lib'].unique())
    })
    return data_processed
def plot_table(data, value_count):
    fig, ax = plt.subplots()
    ax.set_title(f'Actives with Count >= {value_count} and Z Score >= 1')
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText = data.values, colLabels = data.columns,
                     cellLoc = 'center', loc = 'center')
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    makedirs('Figures/activeOverview', exist_ok = True)
    plt.savefig(f'Figures/activeOverview/active_table_{value_count}.png')
    plt.clf()
    return
main()