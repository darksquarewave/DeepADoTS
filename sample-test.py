import glob
import json
import os

import numpy as np
import pandas as pd

from src.algorithms import AutoEncoder, DAGMM, RecurrentEBM, LSTMAD, LSTMED
from src.datasets.pandas import PandasDataset
from src.evaluation import Evaluator

def main():
    eval_json()

def eval_json():
    with open('QueryResult.json') as json_file:
        dict_train = json.load(json_file)
        df = pd.DataFrame.from_records(dict_train)
    seed = np.random.randint(np.iinfo(np.uint32).max, size=1, dtype=np.uint32)[0]
    ds = PandasDataset('temperatur_sonne_calvin', df, ignore=['TYPE', 'LOCATIONID', 'TS'])
    evaluator = Evaluator([ds], get_detectors, seed=seed)
    evaluator.train()

def eval_csv():
    df = pd.read_csv('QueryResult.csv')
    seed = np.random.randint(np.iinfo(np.uint32).max, size=1, dtype=np.uint32)[0]
    ds = PandasDataset('temperatur_sonne_calvin', df, ignore=['TYPE', 'LOCATIONID', 'TS'])
    evaluator = Evaluator([ds], get_detectors, seed=seed)
    evaluator.train()

def get_detectors(seed):
    standard_epochs = 40
    window_size = 13
    dets = [
        AutoEncoder(num_epochs=standard_epochs, seed=seed)
    ]
    return dets

if __name__ == '__main__':
    main()
