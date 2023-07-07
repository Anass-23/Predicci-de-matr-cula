# Common
from pathlib import Path
from collections import namedtuple
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Data
from data.dataset import (
    PrimeraTransformacio, 
    SegonaTransformacio
)
from data.taumat import CarregaTaules

# Shared
from shared.parsing.AssigParser import AssigParser
from shared.classes.AssigPipeline import AssigPipeline
from shared.classes.AssigManager import AssigManager
from shared.visualize.MetricsPlotter import MetricsPlotter
from shared.visualize.tree import plot_assig_tree, prun_assig_tree

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

    # 
    # TODO: Afeggir el módul logging per obtenir en tot moment info dels processos
    #
    print("[INFO] X_test shape:", X_test_pt.shape)
    print("[INFO] y_test shape:", y_test_pt.shape)
    print("[INFO] X_train shape:", X_train_pt.shape)
    print("[INFO] y_train shape:", y_train_pt.shape)
    # print(X_pt, y_pt)
    # raise

    #
    # Experiments
    #
    Experiment = namedtuple('Experiment', ['id', 'transf', 'manager', 'clf'])

    # experiments = [
    #     Experiment(
    #         id='DT(depth=3, T1)',
    #         transf='pt',
    #         manager=AssigManager(), 
    #         clf=DecisionTreeClassifier(
    #             max_depth=3,
    #             random_state=RANDOM_STATE
    #         )
    #     ),
    #     Experiment(
    #         id='DT(depth=3, T2)',
    #         transf='st',
    #         manager=AssigManager(), 
    #         clf=DecisionTreeClassifier(
    #             max_depth=3,
    #             random_state=RANDOM_STATE
    #         )
    #     ),
    #     Experiment(
    #         id='DT(depth=4, T1)',
    #         transf='pt',
    #         manager=AssigManager(), 
    #         clf=DecisionTreeClassifier(
    #             max_depth=4,
    #             random_state=RANDOM_STATE
    #         )
    #     ),
    #     Experiment(
    #         id='DT(depth=4, T2)',
    #         transf='st',
    #         manager=AssigManager(), 
    #         clf=DecisionTreeClassifier(
    #             max_depth=4,
    #             random_state=RANDOM_STATE
    #         )
    #     ),
    #     Experiment(
    #         id='DT(depth=5, T1)',
    #         transf='pt',
    #         manager=AssigManager(), 
    #         clf=DecisionTreeClassifier(
    #             max_depth=5,
    #             random_state=RANDOM_STATE
    #         )
    #     ),
    #     Experiment(
    #         id='DT(depth=5, T2)',
    #         transf='st',
    #         manager=AssigManager(), 
    #         clf=DecisionTreeClassifier(
    #             max_depth=5,
    #             random_state=RANDOM_STATE
    #         )
    #     )
    # ]

    # experiments = [
    #     Experiment(
    #         id=r'\textsc{Dt3t1}',
    #         transf='pt',
    #         manager=AssigManager(), 
    #         clf=DecisionTreeClassifier(
    #             max_depth=3,
    #             random_state=RANDOM_STATE
    #         )
    #     ),
    #     Experiment(
    #         id=r'\textsc{Dt4t1}',
    #         transf='pt',
    #         manager=AssigManager(), 
    #         clf=DecisionTreeClassifier(
    #             max_depth=4,
    #             random_state=RANDOM_STATE
    #         )
    #     ),
    #     Experiment(
    #         id=r'\textsc{Dt5t1}',
    #         transf='pt',
    #         manager=AssigManager(), 
    #         clf=DecisionTreeClassifier(
    #             max_depth=5,
    #             random_state=RANDOM_STATE
    #         )
    #     ),
    # ]

    # experiments = [
    #     Experiment(
    #         id=r'\textsc{Dt3t1}',
    #         transf='pt',
    #         manager=AssigManager(), 
    #         clf=DecisionTreeClassifier(
    #             max_depth=3,
    #             random_state=RANDOM_STATE
    #         )
    #     ),
    #     Experiment(
    #         id=r'\textsc{Dt3t2}',
    #         transf='st',
    #         manager=AssigManager(), 
    #         clf=DecisionTreeClassifier(
    #             max_depth=3,
    #             random_state=RANDOM_STATE
    #         )
    #     ),
    #     Experiment(
    #         id=r'\textsc{Dt4t1}',
    #         transf='pt',
    #         manager=AssigManager(), 
    #         clf=DecisionTreeClassifier(
    #             max_depth=4,
    #             random_state=RANDOM_STATE
    #         )
    #     ),
    #     Experiment(
    #         id=r'\textsc{Dt4t2}',
    #         transf='st',
    #         manager=AssigManager(), 
    #         clf=DecisionTreeClassifier(
    #             max_depth=4,
    #             random_state=RANDOM_STATE
    #         )
    #     ),
    # ]

    experiments = [
        Experiment(
            id=r'\textsc{Dt4t1m}',
            transf='pt',
            manager=AssigManager(), 
            clf=DecisionTreeClassifier(
                max_depth=4,
                random_state=RANDOM_STATE
            )
        ),
        Experiment(
            id=r'\textsc{Dt4t2m}',
            transf='st',
            manager=AssigManager(), 
            clf=DecisionTreeClassifier(
                max_depth=4,
                random_state=RANDOM_STATE
            )
        )
    ]

    experiments = [
        Experiment(
            id=r'\textsc{Dt3t1m}',
            transf='pt',
            manager=AssigManager(), 
            clf=DecisionTreeClassifier(
                max_depth=3,
                random_state=RANDOM_STATE
            )
        ),
        Experiment(
            id=r'\textsc{Dt3t2m}',
            transf='st',
            manager=AssigManager(), 
            clf=DecisionTreeClassifier(
                max_depth=3,
                random_state=RANDOM_STATE
            )
        )
    ]

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
    # for exp in experiments[:2]:
    for exp in experiments[:]:
        if exp.transf == 'pt':
            X_train=X_train_pt
        elif exp.transf == 'st':
            X_train=X_train_st
        else:
            raise ValueError(f'[ERROR] Transf ({exp.transf}) no definida')
        exp.manager.fit(acrlst=acrlst, X_train=X_train)
        print(exp.manager.fit_time)

    # 
    # Visualització
    #
    SHOW = {
        'metrics': True,
        'trees': False
    }
    metrics_plotter = MetricsPlotter()
    
    metrics_plotter.plot_all(
        X_test=X_test_pt if experiments[1].transf == 'pt' else X_test_st,
        assig_manager=experiments[1].manager,
        id=experiments[1].id,
        show=SHOW['metrics']
    )

    metrics_plotter.plot_bars(
        X_test=X_test_pt if experiments[0].transf == 'pt' else X_test_st,
        assig_manager=experiments[0].manager,
        id=experiments[0].id,
        show=SHOW['metrics']
    )
    metrics_plotter.plot_metrics(
        X_test=X_test_pt if experiments[0].transf == 'pt' else X_test_st,
        assig_manager=experiments[0].manager,
        id=experiments[0].id,
        show=SHOW['metrics']
    )

    metrics_plotter.compare_experiments(
        X_test_pt=X_test_pt,
        X_test_st=X_test_st,
        experiments=experiments[:],
        show=SHOW['metrics']
    )

    plot_assig_tree(
        assig_pl=experiments[0].manager['PBN'], 
        transf=experiments[0].transf, 
        show=SHOW['trees']
    )
    plot_assig_tree(
        assig_pl=experiments[1].manager['PBN'], 
        transf=experiments[1].transf, 
        show=SHOW['trees']
    )

    plot_assig_tree(
        assig_pl=experiments[0].manager['F'], 
        transf=experiments[0].transf, 
        show=SHOW['trees']
    )
    plot_assig_tree(
        assig_pl=experiments[1].manager['F'], 
        transf=experiments[1].transf, 
        show=SHOW['trees']
    )

    plot_assig_tree(
        assig_pl=experiments[0].manager['I'], 
        transf=experiments[0].transf, 
        show=SHOW['trees']
    )
    plot_assig_tree(
        assig_pl=experiments[1].manager['I'], 
        transf=experiments[1].transf, 
        show=SHOW['trees']
    )
    
    plot_assig_tree(assig_pl=experiments[0].manager['SS'], transf=experiments[0].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[1].manager['SS'], transf=experiments[1].transf, show=SHOW['trees'])

    # 
    # [Tmp]
    #
    plot_assig_tree(assig_pl=experiments[0].manager['ASSI'], transf=experiments[0].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[0].manager['CSL'], transf=experiments[0].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[0].manager['TP'], transf=experiments[0].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[0].manager['PBN'], transf=experiments[0].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[0].manager['PDS'], transf=experiments[0].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[0].manager['SAR'], transf=experiments[0].transf, show=SHOW['trees'])

    plot_assig_tree(assig_pl=experiments[1].manager['ASSI'], transf=experiments[1].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[1].manager['CSL'], transf=experiments[1].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[1].manager['TP'], transf=experiments[1].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[1].manager['PBN'], transf=experiments[1].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[1].manager['PDS'], transf=experiments[1].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[1].manager['SAR'], transf=experiments[1].transf, show=SHOW['trees'])


    # Compraració profunditat
    # Q1 (F)
    plot_assig_tree(assig_pl=experiments[0].manager['F'], transf=experiments[0].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[1].manager['F'], transf=experiments[1].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[2].manager['F'], transf=experiments[2].transf, show=SHOW['trees'])

    # Q1 (I)
    plot_assig_tree(assig_pl=experiments[0].manager['I'], transf=experiments[0].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[1].manager['I'], transf=experiments[1].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[2].manager['I'], transf=experiments[2].transf, show=SHOW['trees'])

    # Q1 (FMT)
    plot_assig_tree(assig_pl=experiments[0].manager['FMT'], transf=experiments[0].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[1].manager['FMT'], transf=experiments[1].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[2].manager['FMT'], transf=experiments[2].transf, show=SHOW['trees'])
    # OPT (BD)
    plot_assig_tree(assig_pl=experiments[0].manager['BD'], transf=experiments[0].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[1].manager['BD'], transf=experiments[1].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[2].manager['BD'], transf=experiments[2].transf, show=SHOW['trees'])

    # OPT (IU)
    plot_assig_tree(assig_pl=experiments[0].manager['IU'], transf=experiments[0].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[1].manager['IU'], transf=experiments[1].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[2].manager['IU'], transf=experiments[2].transf, show=SHOW['trees'])

    # OPT (SSCI)
    plot_assig_tree(assig_pl=experiments[0].manager['SSCI'], transf=experiments[0].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[1].manager['SSCI'], transf=experiments[1].transf, show=SHOW['trees'])
    plot_assig_tree(assig_pl=experiments[2].manager['SSCI'], transf=experiments[2].transf, show=SHOW['trees'])