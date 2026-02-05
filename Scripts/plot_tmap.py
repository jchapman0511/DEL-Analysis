from time import time
from glob import glob
from os.path import splitext, basename
import pandas as pd
from rdkit.Chem import CanonSmiles, MolFromSmiles
from scipy.stats import rankdata
from numpy import array
from mhfp.encoder import MHFPEncoder
import tmap as tm
from faerun import Faerun
from matplotlib.colors import ListedColormap
from os import makedirs

def main():
    start_time = time()
    files = glob('Data/*.parquet')
    for file in files:
        target = splitext(basename(file))[0]
        print('Working on: ', target)
        data = dataGrab(file)
        tmapPlot(data, target)
    return print('Total Runtime (seconds): ', round(time() - start_time, 2))
def dataGrab(file):
    data_raw = pd.read_parquet(file)
    data_raw[['LIB', 'BB1', 'BB2', 'BB3']] = data_raw['compound'].str.split('-', expand = True)
    data_raw['activity'] = [1 if data_raw['target_zscore'].values[index] >= 1 and
                      data_raw['target_count'].values[index] >= 3 else 0 for
                      index in range(len(data_raw))]
    data_sorted = data_raw.sort_values(by = 'target_count', ascending = False)
    data_validSMILES = data_sorted[data_sorted['SMILES'].notna()]
    data_unique = data_validSMILES.drop_duplicates(subset = 'SMILES')
    data_actives = data_unique[data_unique['activity'] == 1]
    data_inactives = data_unique[data_unique['activity'] == 0].sample(n = 10*len(data_actives), replace = False)
    data = pd.concat([data_actives, data_inactives])
    data['mols'] = data['SMILES'].apply(CanonSmiles).apply(MolFromSmiles)
    data = data[data['mols'].notna()]
    return data
def tmapPlot(data, target):
    # Grab graph elements
    smiles = data['SMILES'].tolist()
    libraries = data['LIB'].tolist()
    bb1s = data['BB1'].tolist()
    bb2s = data['BB2'].tolist()
    bb3s = data['BB3'].tolist()
    counts = data['target_count'].tolist()
    zscores = data['target_zscore'].tolist()
    activities = data['activity'].tolist()
    counts_ranked = rankdata(array(counts) / max(counts)) / len(counts)
    zscores_ranked = rankdata(array(zscores) / max(zscores)) / len(zscores)
    info = zip(smiles, libraries, bb1s, bb2s, bb3s, counts, zscores, activities)
    labels = []
    for smile, library, bb1, bb2, bb3, count, zscore, activity in info:
        labels.append(f'{smile}__{smile}__{library}__{bb1}__{bb2}__{bb3}__{count}__{zscore}__{activity}')
    # Encode the SMILES into MHFP for tree diagram
    mols = data['mols'].tolist()
    encoder = MHFPEncoder()
    fps = []
    for mol in mols:
        fps.append(tm.VectorUint(encoder.encode_mol(mol, min_radius = 0)))
    # Builds and plots tree
    lf = tm.LSHForest(2048, 128)
    lf.batch_add(fps)
    lf.index()
    cfg = tm.LayoutConfiguration()
    cfg.k = 100
    cfg.sl_repeats = 2
    cfg.mmm_repeats = 2
    cfg.node_size = 2
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, config = cfg)
    fae = Faerun(
        clear_color = '#222222', coords = False, view = 'front'
    )
    colormap_activity = ListedColormap(["#222222", '#4B9CD3'], name = 'activity')
    fae.add_scatter(
        'DEL',
        {
            'x': x, 'y': y, 'c':[
                counts, counts_ranked, zscores, zscores_ranked, activities
            ],
            'labels': labels
        },
        shader = 'circle',
        colormap = ['viridis', 'viridis', 'viridis', 'viridis', colormap_activity],
        point_scale = 5,
        has_legend = True,
        selected_labels = ['Structure', 'SMILES', 'Library', 'Building Block 1', 'Building Block 2',
                           'Building Block 3', 'Count', 'Z Score', 'Activity'],
        categorical = [False, False, False, False, True],
        series_title = ['Counts', 'Counts Ranked', 'Z Scores', 'Z Scores Ranked', 'Activity'],
        max_legend_label = [
            str(round(max(counts))),
            str(len(counts_ranked)),
            str(round(max(zscores))),
            str(len(zscores_ranked)),
            'Active'
        ],
        min_legend_label = [ 
            str(round(min(counts))),
            '0',
            str(round(min(zscores))),
            '0',
            'Inactive'    
        ],
        title_index = 2,
        legend_title = ['Metric: Counts', 'Metric: Counts', 
                        'Metric: Z Scores', 'Metric: Z Scores',
                        'Activity']
    )
    fae.add_tree(f'{target}Tree', {'from':s, 'to': t}, point_helper = 'DEL')
    makedirs('Figures/tmap', exist_ok = True)
    fae.plot(f'{target} DEL', template = 'smiles', path = 'Figures/tmap')
    return
main()