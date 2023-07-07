# Common
from pathlib import Path
from collections import OrderedDict, namedtuple
from tqdm import tqdm
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data
from data.dataset import PrimeraTransformacio, SegonaTransformacio
from data.taumat import CarregaTaules

# Shared
from shared.parsing.AssigParser import AssigParser
from shared.classes.AssigPipeline import AssigPipeline
from shared.classes.AssigManager import AssigManager
from shared.visualize.MetricsPlotter import MetricsPlotter
from shared.visualize.tree import plot_assig_tree, prun_assig_tree

# Paths
OUTPUT_DIR_PATH  = './'

BASE_PATH      = Path('../data')
RAW_PATH       = BASE_PATH / 'raw'
INTERIM_PATH   = BASE_PATH / 'interim'
PROCESSED_PATH = BASE_PATH / 'processed'

RAW_MAT_FILE_PATH  = RAW_PATH / 'matricules.anon.csv'
RAW_ACRO_FILE_PATH = RAW_PATH / 'acronims.csv'


if __name__ == '__main__':

    RANDOM_STATE = 123
    assig_parser = AssigParser()
    assig_parser.parse_args()

    # 
    # Càrrega del Dataset
    #
    (tm, ta) = CarregaTaules(nom_mat=RAW_MAT_FILE_PATH,
        nom_acr=RAW_ACRO_FILE_PATH,
        reporta=False
    )
    pt = PrimeraTransformacio()
    st = SegonaTransformacio()

    pt.add_ta(ta)
    st.add_ta(ta)

    (X_pt, y_pt) = pt.load_data(PROCESSED_PATH / 'primerDataset.csv')
    (X_st, y_st) = st.load_data(PROCESSED_PATH / 'segonDataset.csv')
    
    categorical_features  = []
    numerical_features_pt = []
    numerical_features_st = []

    if assig_parser.args.dataset == 'v1':
        print('[INFO] Carregant dataset (Versió 1)')

        # Primera transformació
        X_pt = X_pt.drop(columns=['EDAT', 'VIA', 'ORDRE', 'NACC'])
        X_pt = X_pt.loc[:, ~X_pt.columns.str.endswith('becat')]
        numerical_features_pt   = X_pt.columns.tolist()

        # Segona transformació
        X_st = X_st.drop(columns=['EDAT', 'VIA', 'ORDRE', 'NACC', 'BECAT'])
        numerical_features_st   = X_st.columns.tolist()

        # Comú
        # numerical_features   = X_pt.columns.tolist()

    else:
        print('[INFO] Carregant dataset (Versió 2)')
        
        categorical_features = X_pt[['EDAT', 'VIA', 'ORDRE']].columns.tolist()
        numerical_features_pt   = X_pt.drop(columns=['EDAT', 'VIA', 'ORDRE']).columns.tolist()

        categorical_features  = X_st[['EDAT', 'VIA', 'ORDRE']].columns.tolist()
        numerical_features_st = X_st.drop(columns=['EDAT', 'VIA', 'ORDRE']).columns.tolist()
    
    #
    # Repartiment de les dades
    #
    X_train_pt, X_test_pt, y_train_pt, y_test_pt = train_test_split(
        X_pt,
        y_pt,
        test_size=0.2,
        random_state=42
    )
    
    X_train_st, X_test_st, y_train_st, y_test_st = train_test_split(
        X_st,
        y_st,
        test_size=0.2,
        random_state=42
    )

    print("[INFO] X_test shape:", X_test_pt.shape)
    print("[INFO] y_test shape:", y_test_pt.shape)
    print("[INFO] X_train shape:", X_train_pt.shape)
    print("[INFO] y_train shape:", y_train_pt.shape)

    #
    # Experiments
    #
    Experiment = namedtuple('Experiment', ['id', 'transf', 'manager', 'clf'])

    experiments_dict = {
        'cmp_inicial': [
            Experiment(
                id=r'\textsc{Dt3}',
                manager=AssigManager(),
                transf='pt', 
                clf=DecisionTreeClassifier(
                    max_depth=3,
                    random_state=RANDOM_STATE
                )
            ),
            Experiment(
                id=r'\textsc{Dt4}',
                manager=AssigManager(),
                transf='pt', 
                clf=DecisionTreeClassifier(
                    max_depth=4,
                    random_state=RANDOM_STATE
                )
            ),
            Experiment(
                id=r'\textsc{RfcLog2}',
                manager=AssigManager(),
                transf='pt', 
                clf=RandomForestClassifier(
                    warm_start=False,
                    n_estimators=20,
                    max_depth=4,
                    oob_score=True,
                    max_features="log2",
                    bootstrap=True,
                    random_state=RANDOM_STATE
                )
            ),
            Experiment(
                id=r'\textsc{RftSqrt}',
                manager=AssigManager(),
                transf='pt', 
                clf=RandomForestClassifier(
                    warm_start=False,
                    n_estimators=20,
                    max_depth=4,
                    oob_score=True,
                    max_features="sqrt",
                    bootstrap=True,
                    random_state=RANDOM_STATE
                )
            ),
            Experiment(
                id=r'\textsc{Rft0.5}',
                manager=AssigManager(),
                transf='pt', 
                clf=RandomForestClassifier(
                    warm_start=False,
                    n_estimators=20,
                    max_depth=4,
                    oob_score=True,
                    max_features=0.5,
                    bootstrap=True,
                    random_state=RANDOM_STATE
                )
            ),
            Experiment(
                id=r'\textsc{RftAll}',
                manager=AssigManager(),
                transf='pt', 
                clf=RandomForestClassifier(
                    warm_start=False,
                    n_estimators=20,
                    max_depth=4,
                    oob_score=True,
                    max_features=None,
                    bootstrap=True,
                    random_state=RANDOM_STATE
                )
            )
        ],
        'cmp_estimadors_sqrt': [
            Experiment(
                id=r'\textsc{RftSqrt}',
                manager=AssigManager(),
                transf='pt', 
                clf=RandomForestClassifier(
                    warm_start=False,
                    n_estimators=20,
                    max_depth=4,
                    oob_score=True,
                    max_features="sqrt",
                    bootstrap=True,
                    random_state=RANDOM_STATE
                )
            ),
            Experiment(
                id=r'\textsc{Rft100Sqrt}',
                manager=AssigManager(),
                transf='pt', 
                clf=RandomForestClassifier(
                    warm_start=False,
                    n_estimators=100,
                    max_depth=4,
                    oob_score=True,
                    max_features="sqrt",
                    bootstrap=True,
                    random_state=RANDOM_STATE
                )
            ),
            Experiment(
                id=r'\textsc{Rft200Sqrt}',
                manager=AssigManager(),
                transf='pt', 
                clf=RandomForestClassifier(
                    warm_start=False,
                    n_estimators=200,
                    max_depth=4,
                    oob_score=True,
                    max_features="sqrt",
                    bootstrap=True,
                    random_state=RANDOM_STATE
                )
            ),
            Experiment(
                id=r'\textsc{Rft500Sqrt}',
                manager=AssigManager(),
                transf='pt', 
                clf=RandomForestClassifier(
                    warm_start=False,
                    n_estimators=500,
                    max_depth=4,
                    oob_score=True,
                    max_features="sqrt",
                    bootstrap=True,
                    random_state=RANDOM_STATE
                )
            ),
        ],
        'cmp_confs_optims': [
            Experiment(
                id=r'\textsc{RftConf1}',
                manager=AssigManager(),
                transf='pt', 
                clf=RandomForestClassifier(
                    warm_start=True,
                    n_estimators=10,
                    max_depth=4,
                    oob_score=True,
                    max_features=0.5,
                    max_samples=0.8,
                    bootstrap=True,
                    random_state=RANDOM_STATE
                )
            ),
            Experiment(
                id=r'\textsc{RftConf2}',
                manager=AssigManager(),
                transf='pt', 
                clf=RandomForestClassifier(
                    warm_start=True,
                    n_estimators=20,
                    max_depth=4,
                    oob_score=True,
                    max_features=0.5,
                    max_samples=0.8,
                    bootstrap=True,
                    random_state=RANDOM_STATE
                )
            ),
            Experiment(
                id=r'\textsc{RftConf3}',
                manager=AssigManager(),
                transf='pt', 
                clf=RandomForestClassifier(
                    warm_start=True,
                    n_estimators=10,
                    max_depth=4,
                    oob_score=True,
                    max_features=0.8,
                    max_samples=0.8,
                    bootstrap=True,
                    random_state=RANDOM_STATE
                )
            ),
            Experiment(
                id=r'\textsc{RftConf4}',
                manager=AssigManager(),
                transf='pt', 
                clf=RandomForestClassifier(
                    warm_start=True,
                    n_estimators=20,
                    max_depth=4,
                    oob_score=True,
                    max_features=0.8,
                    max_samples=0.8,
                    bootstrap=True,
                    random_state=RANDOM_STATE
                )
            ),
        ],
        'conf_definitiu': [
            Experiment(
                id=r'\textsc{RftConf1}',
                manager=AssigManager(),
                transf='pt', 
                clf=RandomForestClassifier(
                    warm_start=True,
                    n_estimators=15,
                    max_depth=3,
                    oob_score=True,
                    max_features=0.5,
                    max_samples=0.8,
                    bootstrap=True,
                    random_state=RANDOM_STATE
                )
            ),
            Experiment(
                id=r'\textsc{RftConf2}',
                manager=AssigManager(),
                transf='st', 
                clf=RandomForestClassifier(
                    warm_start=True,
                    n_estimators=15,
                    max_depth=3,
                    oob_score=True,
                    max_features=0.5,
                    max_samples=0.8,
                    bootstrap=True,
                    random_state=RANDOM_STATE
                )
            ),
        ],
        "tmp_heatmap": [
            Experiment(
                id=r'\textsc{RftConf2}',
                manager=AssigManager(),
                transf='st', 
                clf=RandomForestClassifier(
                    warm_start=True,
                    n_estimators=15,
                    max_depth=3,
                    oob_score=True,
                    max_features=0.5,
                    max_samples=0.8,
                    bootstrap=True,
                    random_state=RANDOM_STATE
                )
            ),
        ]
    }
    
    #
    # Selecció dels experiments
    #
    # 
    # experiments = experiments_dict['cmp_confs_optims']
    # experiments = experiments_dict['conf_definitiu']
    experiments = experiments_dict['tmp_heatmap']

    #
    # Càrrega dels models
    #
    acrlst =  assig_parser.acrlst if assig_parser.acrlst else [acr for acr in ta.get_acrlst() if acr]
    for acr in acrlst:
        for exp in experiments:
            if exp.transf == 'pt':
                y_train, y_test = y_train_pt[acr], y_test_pt[acr]
                numerical_features = numerical_features_pt
            elif exp.transf == 'st':
                y_train, y_test = y_train_st[acr], y_test_st[acr]
                numerical_features = numerical_features_st
            else:
                raise ValueError(f'[ERROR] Transf ({exp.transf}) no definida')
            
            exp.manager.add_model(
                acr=acr,
                model=AssigPipeline(
                    id=acr,
                    y_train=y_train,
                    y_test=y_test,
                    categorical_features=categorical_features,
                    numerical_features=numerical_features,
                    clf=clone(exp.clf)
                )
            )
    
    # 
    # Entrenament
    #
    for exp in experiments[:]:
        if exp.transf == 'pt':
            X_train=X_train_pt
        elif exp.transf == 'st':
            X_train=X_train_st
        else:
            raise ValueError(f'[ERROR] Transf ({exp.transf}) no definida')
        exp.manager.fit(acrlst=acrlst, X_train=X_train)
        print(exp.manager.fit_time)

    # import pickle
    # for acr in acrlst:
    #     with open(f'./models/{acr}-{exp.transf}.pkl', 'wb') as f:
    #         pickle.dump(exp.manager[acr].clf, f)
        
    #     print(exp.manager[acr].clf)


    acrlst_group = [
        ['MBE', 'F', 'I', 'ISD', 'FMT'],
        ['ES', 'TCO1', 'TP', 'SD', 'TCI'],
        ['MAE', 'TCO2', 'DP', 'EM', 'CSL'],
        ['SA', 'PBN', 'ACO', 'CSR', 'SS'],
        ['PCTR', 'GOP', 'SO', 'XC', 'PDS'],
        ['SEN', 'ESI', 'ASSI', 'SEC', ''],
        ['IS', 'SAR', '', '', ''],
        ['TFG', '', '', '', '']
    ][::-1]

    quads = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8'][::-1]

    importances = []
    for acrlst_g in acrlst_group:
        tmp = []
        for acr_target in acrlst_g:
            if acr_target:
                rel_featimp_list = []
                for acr in acrlst:                    
                    for lbl, featimp in zip(X_st.columns, experiments[0].manager[acr].clf.feature_importances_):
                        if lbl.startswith(acr_target):
                            rel_featimp_list.append((lbl, featimp))

                featimp_avg = sum([feat_imp for _, feat_imp in rel_featimp_list]) / len(rel_featimp_list)
                tmp.append(featimp_avg)


            else: 
                tmp.append(0)
        
        importances.append(tmp)

    print(importances)


    # HeatMap
    sns.set()
    ax = sns.heatmap(importances, annot=False, cmap='plasma')

    ax.set_xticklabels([], rotation=90)
    ax.set_yticklabels(quads, rotation=0)

    for i in range(len(acrlst_group)):
        for j in range(len(acrlst_group[i])):
            label = acrlst_group[i][j]
            x = j
            y = i
            ax.annotate(label, xy=(x, y), xytext=(x+0.5, y+0.5),
                        color='white', fontsize=10, ha='center', va='center', weight='bold')

    # Legend
    cbar = ax.collections[0].colorbar
    cbar.set_label(r'\emph{Feature Importance}')

    plt.xlabel('Assignatura')
    plt.ylabel('Quadrimestre')
    plt.title('Importància de cada assignatura')
    plt.savefig('feat_imp.png')
    plt.savefig('feat_imp.pgf')

    
    plt.show()
    
    # 
    # Visualització
    #
    # SHOW = {
    #     'metrics': True,
    #     'trees': False
    # }
    # metrics_plotter = MetricsPlotter()
    
    # metrics_plotter.compare_experiments(
    #     X_test_pt=X_test_pt,
    #     X_test_st=X_test_st,
    #     experiments=experiments[:],
    #     show=SHOW['metrics']
    # )

    # metrics_plotter.plot_all(
    #     X_test=X_test_pt if experiments[1].transf == 'pt' else X_test_st,
    #     assig_manager=experiments[1].manager,
    #     id=experiments[1].id,
    #     show=SHOW['metrics']
    # )

    # def plot_random_forest_trees(forest):
    #     fig, axes = plt.subplots(nrows=forest.n_estimators // 2, ncols=2, figsize=(10, 20))
    #     axes = axes.flatten()

    #     for index, tree in enumerate(forest.estimators_):
    #         axes[index].set_title(f'Tree {index+1}')
    #         plot_tree(tree, ax=axes[index])

    #     plt.tight_layout()
    #     fig.savefig('test.png')
    #     fig.savefig('test.pgf')
    #     plt.show()

    # # plot_random_forest_trees(forest=experiments[1].manager['PBN'].pl.named_steps['clf'])
    
    # plot_assig_tree(experiments[1].manager['PBN'], transf='st', show=True)
    # plot_assig_tree(experiments[1].manager['PBN'], transf='st', show=True)
    # raise



    #
    # Evolució OOB-score dels [Boscos]
    #

    models = {acr: None for acr in acrlst}

    plt.figure(figsize=(10, 5), dpi=150)
    # for acr in tqdm(acrlst[15:20], desc='OOB-scores'):
    for acr in tqdm(acrlst[:], desc='OOB-scores'):

        ensemble_clfs = [
            (
                'RftConf2',
                RandomForestClassifier(
                    warm_start=True,
                    oob_score=True,
                    max_features=0.5,
                    max_samples=0.8,
                    bootstrap=True,
                    random_state=RANDOM_STATE
                ),
            )
        ]

        best_error_rate = 100
        best_n_estimators = None
        min_estimators = 5
        max_estimators = 100

        X_train, y_train = X_train_st, y_train_st

        for label, clf in ensemble_clfs:
            error_rate = []
            for i in range(min_estimators, max_estimators + 1, 5):
                clf.set_params(n_estimators=i)
                clf.fit(X_train, y_train[acr])
                oob_error = 1 - clf.oob_score_
                error_rate.append((i, oob_error))

            models[acr] = {'error_rate': error_rate, 'best_n_estimators': best_n_estimators}
            models[acr] = {'error_rate': error_rate}
            best_n_estimators = min(error_rate, key=lambda x: x[1])[0]
            models[acr]['best_n_estimators'] = best_n_estimators

            print('conf:', label)
            print('error_rate:', error_rate)
            print('best_n_estimators:', best_n_estimators)
            print()

        xs, ys = zip(*models[acr]['error_rate'])
        plt.plot(xs, ys, label=acr)

    plt.xlim(min_estimators, max_estimators)
    plt.xticks(range(0, 100, 10))
    plt.grid(True)
    plt.xlabel("Número d'estimadors")
    plt.ylabel("OOB error")
    plt.legend(loc="upper right")
    plt.savefig('oob-error.png')
    plt.savefig('oob-error.pgf')
    plt.show()

    acrslabels = []
    best_estimators = []

    for acr, model in models.items():
        if model:
            acrslabels.append(acr)
            best_estimators.append(model['best_n_estimators'])

    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(acrslabels, best_estimators, color='red', marker='o', linestyle='--', alpha=0.8)
    plt.xlabel("Assignatura")
    plt.xticks(rotation=90)
    plt.ylabel("Millor número d'estimadors")
    plt.tight_layout()

    plt.savefig('best.png')
    plt.savefig('best.pgf')
    plt.show()


    for acr, model in models.items():
        print(acr, model)

    raise
    
    experiments = [
        Experiment(
            id=r'\textsc{RftConf2}',
            manager=AssigManager(),
            transf='st', 
            clf=RandomForestClassifier(
                warm_start=True,
                # n_estimators=15,
                max_depth=3,
                oob_score=True,
                max_features=0.5,
                max_samples=0.8,
                bootstrap=True,
                random_state=RANDOM_STATE
            )
        )
    ]


    acrlst =  assig_parser.acrlst if assig_parser.acrlst else [acr for acr in ta.get_acrlst() if acr]
    for acr in acrlst:
        for exp in experiments:
            if exp.transf == 'pt':
                y_train, y_test = y_train_pt[acr], y_test_pt[acr]
                numerical_features = numerical_features_pt
            elif exp.transf == 'st':
                y_train, y_test = y_train_st[acr], y_test_st[acr]
                numerical_features = numerical_features_st
            else:
                raise ValueError(f'[ERROR] Transf ({exp.transf}) no definida')
            
            clf_clone = clone(exp.clf)
            clf_clone.n_estimators = models[acr]['best_n_estimators']
            print(acr, '->', models[acr]['best_n_estimators'])
            print(clf_clone)
            exp.manager.add_model(
                acr=acr,
                model=AssigPipeline(
                    id=acr,
                    y_train=y_train,
                    y_test=y_test,
                    categorical_features=categorical_features,
                    numerical_features=numerical_features,
                    clf=clone(exp.clf)
                )
            )

    for exp in experiments[:]:
        if exp.transf == 'pt':
            X_train=X_train_pt
        elif exp.transf == 'st':
            X_train=X_train_st
        else:
            raise ValueError(f'[ERROR] Transf ({exp.transf}) no definida')
        exp.manager.fit(acrlst=acrlst, X_train=X_train)
        print(exp.manager.fit_time)


    SHOW = {
        'metrics': True,
        'trees': False
    }
    metrics_plotter = MetricsPlotter()
    
    metrics_plotter.plot_all(
        X_test=X_test_pt if experiments[0].transf == 'pt' else X_test_st,
        assig_manager=experiments[0].manager,
        id=experiments[0].id,
        show=SHOW['metrics']
    )

    # metrics_plotter.compare_experiments(
    #     X_test_pt=X_test_pt,
    #     X_test_st=X_test_st,
    #     experiments=experiments[:],
    #     show=SHOW['metrics']
    # )


    #
    # Feature Importances
    # 
    
    acrlst_group = [
            ['MBE', 'F', 'I', 'ISD', 'FMT'],
            ['ES', 'TCO1', 'TP', 'SD', 'TCI'],
            ['MAE', 'TCO2', 'DP', 'EM', 'CSL'],
            ['SA', 'PBN', 'ACO', 'CSR', 'SS'],
            ['PCTR', 'GOP', 'SO', 'XC', 'PDS'],
            ['SEN', 'ESI', 'ASSI', 'SEC', ''],
            ['IS', 'SAR', '', '', ''],
            ['TFG', '', '', '', '']
    ][::-1]
    quads = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8'][::-1]

    importances = []
    for acrlst_g in acrlst_group:
        tmp = []
        for acr in acrlst_g:
            print(acr, 'XD')
            if acr:
                tmp.append(
                    experiments[0].manager[acr].clf.feature_importances_[0]
                )
            else: 
                tmp.append(0)
        importances.append(tmp)

    print(importances)


    # HeatMap
    sns.set()
    ax = sns.heatmap(importances, annot=False, cmap='plasma')

    ax.set_xticklabels([], rotation=90)
    ax.set_yticklabels(quads, rotation=0)

    for i in range(len(acrlst_group)):
        for j in range(len(acrlst_group[i])):
            label = acrlst_group[i][j]
            x = j
            y = i
            ax.annotate(label, xy=(x, y), xytext=(x+0.5, y+0.5),
                        color='black', fontsize=10, ha='center', va='center')
    
    # Assignatura Target
    rect = plt.Rectangle((1, 4), 1, 1, fill=False, edgecolor='red', lw=2, alpha=0.5) # PBN
    ax.add_patch(rect)

    # Legend
    cbar = ax.collections[0].colorbar
    cbar.set_label('Importance')

    plt.xlabel('Assignatura')
    plt.ylabel('Quadrimestre')
    plt.title('Feature Importances Heatmap')
    plt.show()


    raise

    importances = models['PBN'].clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    k = 20
    top_k_features = X_st.columns[indices][:k]

    plt.bar(top_k_features, importances[indices][:k])
    plt.xticks(rotation='vertical')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Top {} Most Important Features'.format(k))
    plt.show()
    





    raise


    models = {acr: None for acr in acrlst}
    print (models)

    for acr in tqdm(acrlst[:1], desc='OOB-scores'):
        
        ensemble_clfs = [
            (
                'RftConf1',
                RandomForestClassifier(
                    warm_start=True,
                    oob_score=True,
                    max_features=0.5,
                    max_samples=0.8,
                    bootstrap=True,
                    random_state=RANDOM_STATE
                ),
            ),
            (
                'RftConf2',
                RandomForestClassifier(
                    warm_start=True,
                    oob_score=True,
                    max_features=0.8,
                    max_samples=0.8,
                    bootstrap=True,
                    random_state=RANDOM_STATE
                ),
            ),
        ]
        
        error_rate     = OrderedDict((label, []) for label, _ in ensemble_clfs)
        min_estimators = 10
        max_estimators = 150
        
        X_train, y_train = X_train_st, y_train_st

        for label, clf in ensemble_clfs:
            for i in range(min_estimators, max_estimators + 1, 5):
                clf.set_params(n_estimators=i)
                # clf.fit(X_train, y_train['PBN'])
                clf.fit(X_train, y_train[acr])

                # Record the OOB error for each `n_estimators=i` setting.
                oob_error = 1 - clf.oob_score_
                error_rate[label].append((i, oob_error))

        models[acr] = error_rate

        # for model in models:
         #   ordered_dict = models[model]
        for label, clf in ensemble_clfs:
            print('conf:', label)
            print('error_rate:', error_rate)
            print()
        

        # for exp in models[model]:
        #     print(exp)
        #     for ensemble_clf, error_rate in exp:
        #         print(ensemble_clf, error_rate)



    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)


    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()