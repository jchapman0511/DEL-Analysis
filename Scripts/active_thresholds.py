from time import time
from glob import glob
from os.path import basename, splitext
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main(count = 5, threshold_zscore = 1, threshold_polyo = 4):
    start_time = time()
    files = glob('Data/*.parquet')
    for file in files:
        zscore = 0
        polyo = 0
        target = splitext(basename(file))[0].split('_')[0]
        print('Working on:', target)
        values_zscore = []
        values_polyo = []
        activeCount_zscore = []
        activeCount_polyo = []
        while zscore <= threshold_zscore:
            actives_zscore, actives_polyo = dataGrab(file, count, zscore, polyo)
            activeCount_zscore.append(actives_zscore)
            activeCount_polyo.append(actives_polyo)
            values_zscore.append(zscore)
            zscore += threshold_zscore / 8
            values_polyo.append(polyo)
            polyo += threshold_polyo / 8
        data_zscore = pd.DataFrame({'Z Scores': values_zscore, 'Active Count': activeCount_zscore, 'Target': target})
        data_polyo = pd.DataFrame({'Poly O': values_polyo, 'Active Count': activeCount_polyo, 'Target': target})
        activePlot(data_zscore, 'Z Scores', 'Active Count', target, 'zscore')
        activePlot(data_polyo, 'Poly O', 'Active Count', target, 'polyo')
    return print('Total Runtime (seconds): ', round(time() - start_time, 2))
def dataGrab(file, count, zscore, polyo):
    data = pd.read_parquet(file)
    data = data[data['SMILES'].notna()]
    data_agg = data.groupby('SMILES').agg({
        'target_count': 'sum', 'target_zscore': 'mean', 'polyo': 'mean'
        })
    data_agg['zscore_activity'] = [1 if  data_agg['target_count'].values[index] >= count and
                      data_agg['target_zscore'].values[index] >= zscore else 0 for
                      index in range(len(data_agg))]
    data_agg['polyo_activity'] = [1 if  data_agg['target_count'].values[index] >= count and
                      data_agg['polyo'].values[index] >= polyo else 0 for
                      index in range(len(data_agg))]
    return len(data_agg[data_agg['zscore_activity'] == 1]), len(data_agg[data_agg['polyo_activity'] == 1])
def activePlot(data, x_title, y_title, target, method):
    sns.scatterplot(data = data, x = x_title, y = y_title)
    plt.savefig(f'Figures/activeOverview/activeDistribution_{target}_{method}.png')
    plt.clf()
    return
main()