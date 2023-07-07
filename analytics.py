# Common
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Data
from data.dataset import PrimeraTransformacio
from data.taumat import CarregaTaules

# Shared
OUTPUT_DIR_PATH  = './'

BASE_PATH      = Path('../data')
RAW_PATH       = BASE_PATH / 'raw'
INTERIM_PATH   = BASE_PATH / 'interim'
PROCESSED_PATH = BASE_PATH / 'processed'

RAW_MAT_FILE_PATH  = RAW_PATH / 'matricules.anon.csv'
RAW_ACRO_FILE_PATH = RAW_PATH / 'acronims.csv'


def matriculacions():
    (tm, ta) = CarregaTaules(nom_mat=RAW_MAT_FILE_PATH,
        nom_acr=RAW_ACRO_FILE_PATH,
        reporta=False
    )
    pt = PrimeraTransformacio()
    pt.add_ta(ta)

    (X, y) = pt.load_data(PROCESSED_PATH / 'primerDataset.csv')

    acrlst = [
        'MBE', 'F', 'I', 'ISD', 'FMT', 
        'ES', 'TCO1', 'TP', 'SD', 'TCI', 
        'MAE', 'TCO2', 'DP', 'EM', 'CSL', 
        'SA', 'PBN', 'ACO', 'CSR', 'SS', 
        'PCTR', 'GOP', 'SO', 'XC', 'PDS', 
        'SEN', 'ESI', 'ASSI', 'SEC', 
        'IS', 'SAR',
        'TFG',
        'MIC', 'SC', 'SSCI', 'AE', 'GQSIQSMA', 'BD', 'IU', 'RE'
        # 'Q', 'EG', 'CTM', 'SM', 'RM', 'RE', 'GQSIQSMA', 'AE', 'SC', 'SSCI', 'IU', 'MIC', 'BD',
    ]

    colors = ['red']*5 + ['green']*5 + ['blue']*5 + ['darkorange']*5 + ['purple'] *5 + ['gray']*4 + ['magenta']*2 + ['darkorange'] + ['chartreuse']*8

    # print(X.columns.tolist())
    mats = {acr: y[acr].sum() for acr in acrlst}
    # print(mats)

    x = np.arange(len(mats))
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.bar(x, mats.values(), color=colors, alpha=0.9)

    ax.set_ylabel('Matriculacions')
    ax.set_xlabel('Assignatura')
    ax.set_xticks(x)
    ax.set_xticklabels(mats.keys(), rotation=90)

    plt.tight_layout()
    plt.savefig('alytic_1.png')
    plt.savefig('alytic_1.pgf')
    plt.show()


def diff_metrics(csv1, csv2):
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.compare.html
    df1 = pd.read_csv(csv1, index_col=0)
    df2 = pd.read_csv(csv2, index_col=0)

    # print(df1)
    diff_metrics = df1.compare(df2).fillna(0.0)
    print(diff_metrics)
    diff_metrics.to_csv('tmp2/diff.csv')


if __name__ == '__main__':
    # matriculacions()
    
    # csv1 = '../reports/data/Exp1/v1/metrics_results.csv'
    # csv2 = '../reports/data/Exp1/v2/metrics_results.csv'
    
    # csv1 = r'tmp2/metrics/metrics_results_Dt3t1.csv'
    # csv2 = r'tmp2/metrics/metrics_results_Dt3t1m.csv'
    # diff_metrics(csv1, csv2)

    csv1 = r'tmp2/metrics/metrics_results_Dt3t2.csv'
    csv2 = r'tmp2/metrics/metrics_results_Dt3t2m.csv'
    diff_metrics(csv1, csv2)