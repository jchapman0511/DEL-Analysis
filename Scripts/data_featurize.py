from time import time
import argparse
from glob import glob
import pandas as pd
from os.path import splitext, basename
from rdkit.Chem import CanonSmiles, MolFromSmiles
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from os import makedirs

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_type', type = str)
parser.add_argument('-l', '--labeled', type = bool)
parser.add_argument('-c', '--count', type = int)
parser.add_argument('-z', '--zscore', type = float)
parser.add_argument('-m', '--method', type = str)
args = parser.parse_args()

def featurize(file_type = args.file_type,
         labeled = args.labeled,
         count = args.count,
         zscore = args.zscore,
         method = args.method):
    t0 = time()
    count = count if count is not None else 5 # Default count if None passed
    makedirs('Data/featurized', exist_ok = True)
    files = glob(f'Data/primary/*.{file_type}')
    for file in files:
        target = splitext(basename(file))[0]
        print('Processing:', target)
        data = file_types.get(file_type)(file)
        if labeled:
            try:
                data = pd.DataFrame({'SMILES': data['SMILES'],
                                     'activity': data['activity']})
            except KeyError:
                print('Please include "SMILES" and "activity" columns...')
                continue
        else:
            data = dataGrab(data, count, zscore)
        data_features = feature_methods.get(method)(data)
        data_features.to_csv(f'Data/featurized/{target}_{method}.csv')
    return print(f'Finished in: {round(time() - t0, 2)} seconds')

def dataGrab(data, count:int = 5, zscore:float = None):
        data = data.sort_values(by = 'target_count', ascending = False)
        data = data[data['SMILES'].notna()].drop_duplicates(subset = 'SMILES')
        if zscore == None:
            data['activity'] = [1 if data['target_count'].values[index] >= count
                                else 0 for index in range(len(data))]
        elif zscore != None:
            data['activity'] = [1 if data['target_count'].values[index] >= count
                                and data['target_zscore'].values[index] >= zscore
                                else 0 for index in range(len(data))]
        data_actives = data[data['activity'] == 1]
        data_inactives = data[data['activity'] == 0].sample(
            n = 10*len(data_actives), random_state = 42
        )
        data = pd.concat([data_actives, data_inactives], ignore_index = True)
        data_activity = pd.DataFrame({
            'SMILES': data['SMILES'], 'activity': data['activity']
        })
        print('Active Count:', len(data[data['activity'] == 1]))
        return data_activity

def ECFP4(data, n_bits:int = 2048):
    data['CanonSmiles'] = data['SMILES'].apply(CanonSmiles)
    mols = data['CanonSmiles'].apply(MolFromSmiles)
    morganGen = GetMorganGenerator(radius = 2, fpSize = n_bits)
    data_fp = pd.DataFrame([list(fp) for fp in morganGen.GetFingerprints(mols)])
    data_fp['activity'] = data['activity']
    return data_fp

feature_methods = {
    'ECFP4': ECFP4
}
file_types = {
    'csv': pd.read_csv,
    'parquet': pd.read_parquet
}

featurize()