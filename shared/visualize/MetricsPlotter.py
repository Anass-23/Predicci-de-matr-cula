import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Any
from sklearn.metrics import recall_score, precision_score, f1_score
from shared.classes.AssigManager import AssigManager
from shared.visualize.config import BASE_TMP_PATH, METRICS_TMP_PATH, PLOTS_TMP_PATH

class MetricsPlotter:
    
    def plot_all(
            self, 
            id: str,
            X_test: pd.DataFrame,
            assig_manager: AssigManager,
            metric: str = 'all',
            annotate: bool = False,
            show: bool = False
        ):
        
        metrics = {
            'recall':    ('Recall' ,  []),
            'precision': ('Precisió', []),
            'f1_score':  ('F-score', []),
        }
        
        metric_results = pd.DataFrame(columns=['Assignatura', 'Recall', 'Precisió', 'F-score'])
        c_test = []
        c_pred = []

        for acr in assig_manager.fitted_acr_models:
            c_test.append(len([e for e in assig_manager[acr].y_test if e]))
            # NOTE: !!!!!! MOTIU PRINCIPAL DEL PROBLEMA !!!!!!!
            # c_pred.append(len([e for e in assig_manager[acr].clf.predict(X_test) if e]))
            c_pred.append(len([e for e in assig_manager[acr].pl.predict(X_test) if e]))
            
        x = np.arange(len(assig_manager.fitted_acr_models))
        width = 0.35

        fig, ax = plt.subplots(2, figsize=(10, 6), dpi=150)
        ax[0].bar(x-width/2, c_test, width, label=r'y\_test')
        ax[0].bar(x+width/2, c_pred, width, label=r'y\_pred')

        if annotate:
            for container in ax[0].containers:
                ax[0].bar_label(container)
        
        ax[0].set_title('Avaluació del model')
        ax[0].set_ylabel('Matriculacions')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(assig_manager.fitted_acr_models, rotation=90)
        ax[0].legend(loc='upper right')

        for acr in assig_manager.fitted_acr_models:
            y_test = assig_manager[acr].y_test
            y_pred = assig_manager[acr].pl.predict(X_test)

            metrics['recall'][1].append(recall_score(y_test, y_pred, average='binary', zero_division=0))
            metrics['precision'][1].append(precision_score(y_test, y_pred, average='binary', zero_division=0))
            metrics['f1_score'][1].append(f1_score(y_test, y_pred, average='binary', zero_division=0))

            metric_results = metric_results.append({
                'Assignatura': acr,
                'Recall': format(recall_score(y_test, y_pred, average='binary', zero_division=0), '.3f'),
                'Precisió': format(precision_score(y_test, y_pred, average='binary', zero_division=0), '.3f'),
                'F-score': format(f1_score(y_test, y_pred, average='binary', zero_division=0), '.3f')
            }, ignore_index=True)

            print(f'[{acr}]',
                  recall_score(y_test, y_pred, average='binary', zero_division=0),
                  precision_score(y_test, y_pred, average='binary', zero_division=0),
                  f1_score(y_test, y_pred, average='binary', zero_division=0)
            )

        if metric == 'all':
            ax[1].plot(range(0, len(x)), metrics['recall'][1],    color='purple', marker='o', alpha=0.5, label=metrics['recall'][0])
            ax[1].plot(range(0, len(x)), metrics['precision'][1], color='green',  marker='o', alpha=0.5, label=metrics['precision'][0])
            ax[1].plot(range(0, len(x)), metrics['f1_score'][1],  color='red',    marker='o', alpha=0.5, label=metrics['f1_score'][0])

        elif metric in ['recall', 'precision', 'f1_score']:
            ax[1].plot(range(0, len(x)), metrics[metric][1], color='red', marker='o', alpha=0.5, label=metrics[metric][0])

        ax[1].axhline(y=0.8, color='purple', alpha=0.5, lw=1, linestyle='--')

        ax[1].set_ylabel('Proporció')
        ax[1].set_xlabel('Assignatura')
        ax[1].set_yticks(np.arange(0, 1.2, 0.2))
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(assig_manager.fitted_acr_models, rotation=90)
        ax[1].legend(loc='upper right')

        plt.tight_layout()

        # plt.savefig(Path(__file__).parent / '../../tmp/plots/eval.png')
        # plt.savefig(Path(__file__).parent / '../../tmp/plots/eval.pgf')

        plt.savefig(PLOTS_TMP_PATH / f'eval_all_{id}.png')
        plt.savefig(PLOTS_TMP_PATH / f'eval_all_{id}.pgf')

        # metric_results.to_csv(Path(__file__).parent / '../../tmp/metrics/metrics_results.csv', index=False)
        metric_results.to_csv(METRICS_TMP_PATH / f'metrics_results_{id}.csv', index=False)

        if show: 
            plt.show()
        else:
            plt.close()
    
    def plot_bars(
            self, 
            id: str,
            X_test: pd.DataFrame,
            assig_manager: AssigManager,
            annotate: bool = False,
            show: bool = False
        ):
        
        c_test = []
        c_pred = []

        for acr in assig_manager.fitted_acr_models:
            c_test.append(len([e for e in assig_manager[acr].y_test if e]))
            # NOTE: !!!!!! MOTIU PRINCIPAL DEL PROBLEMA !!!!!!!
            # c_pred.append(len([e for e in assig_manager[acr].clf.predict(X_test) if e]))
            c_pred.append(len([e for e in assig_manager[acr].pl.predict(X_test) if e]))
            
        x = np.arange(len(assig_manager.fitted_acr_models))
        width = 0.35

        fig, ax = plt.subplots(1, figsize=(10, 4), dpi=150)
        ax.bar(x-width/2, c_test, width, label=r'y\_test')
        ax.bar(x+width/2, c_pred, width, label=r'y\_pred')

        if annotate:
            for container in ax.containers:
                ax.bar_label(container)
        
        ax.set_title('Avaluació del model')
        ax.set_ylabel('Matriculacions')
        ax.set_xlabel('Assignatura')
        ax.set_xticks(x)
        ax.set_xticklabels(assig_manager.fitted_acr_models, rotation=90)
        ax.legend(loc='upper right')

        plt.tight_layout()

        # plt.savefig(Path(__file__).parent / '../../tmp/plots/eval.png')
        # plt.savefig(Path(__file__).parent / '../../tmp/plots/eval.pgf')

        plt.savefig(PLOTS_TMP_PATH / f'eval_bars{id}.png')
        plt.savefig(PLOTS_TMP_PATH / f'eval_bars{id}.pgf')

        if show: 
            plt.show()
        else:
            plt.close()

    def plot_metrics(
            self, 
            id: str,
            X_test: pd.DataFrame,
            assig_manager: AssigManager,
            metric: str = 'f1_score',
            annotate: bool = False,
            show: bool = False
        ):
    
        metrics = {
            'recall': ('Recall', []),
            'precision': ('Precisió', []),
            'f1_score': ('F-score', []),
        }
        
        metric_results = pd.DataFrame(columns=['Assignatura', 'Recall', 'Precisió', 'F-score'])
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        x = np.arange(len(assig_manager.fitted_acr_models))

        
        for acr in assig_manager.fitted_acr_models:
            y_test = assig_manager[acr].y_test
            y_pred = assig_manager[acr].pl.predict(X_test)

            metrics['recall'][1].append(recall_score(y_test, y_pred, average='binary', zero_division=0))
            metrics['precision'][1].append(precision_score(y_test, y_pred, average='binary', zero_division=0))
            metrics['f1_score'][1].append(f1_score(y_test, y_pred, average='binary', zero_division=0))

            metric_results = metric_results.append({
                'Assignatura': acr,
                'Recall': format(recall_score(y_test, y_pred, average='binary', zero_division=0), '.3f'),
                'Precisió': format(precision_score(y_test, y_pred, average='binary', zero_division=0), '.3f'),
                'F-score': format(f1_score(y_test, y_pred, average='binary', zero_division=0), '.3f')
            }, ignore_index=True)

            print(f'[{acr}]',
                recall_score(y_test, y_pred, average='binary', zero_division=0),
                precision_score(y_test, y_pred, average='binary', zero_division=0),
                f1_score(y_test, y_pred, average='binary', zero_division=0)
            )

        if metric == 'all':
            ax.plot(range(0, len(x)), metrics['recall'][1], color='purple', marker='o', alpha=0.5, label=metrics['recall'][0])
            ax.plot(range(0, len(x)), metrics['precision'][1], color='green', marker='o', alpha=0.5, label=metrics['precision'][0])
            ax.plot(range(0, len(x)), metrics['f1_score'][1], color='red', marker='o', alpha=0.5, label=metrics['f1_score'][0])

        elif metric in ['recall', 'precision', 'f1_score']:
            ax.plot(range(0, len(x)), metrics[metric][1], color='red', marker='o', alpha=0.5, label=metrics[metric][0])

        ax.axhline(y=0.8, color='purple', alpha=0.5, lw=1, linestyle='--')

        ax.set_title('Avaluació del model')
        ax.set_xticks(x)
        ax.set_xticklabels(assig_manager.fitted_acr_models, rotation=90)
        ax.legend(loc='upper right')

        ax.set_ylabel('Proporció')
        ax.set_xlabel('Assignatura')
        ax.set_yticks(np.arange(0, 1.2, 0.2))
        ax.set_xticks(range(0, len(x)))
        ax.set_xticklabels(assig_manager.fitted_acr_models, rotation=90)
        ax.legend(loc='upper right')
        
        plt.tight_layout()

        metric_results.to_csv(METRICS_TMP_PATH / f'metrics_results_{id}.csv', index=False)
        plt.savefig(PLOTS_TMP_PATH / f'eval_metrics{id}.png')
        plt.savefig(PLOTS_TMP_PATH / f'eval_metrics{id}.pgf')
        
        if show: 
            plt.show()
        else:
            plt.close()
                    

    def compare_experiments(
            self, 
            experiments: List[Any], 
            X_test_pt: pd.DataFrame,
            X_test_st: pd.DataFrame,
            title: str = 'Impacte de la profunditat de l\'arbre',
            show: bool = False,
            display_fit_time: bool = True
        ):

        if not display_fit_time:
            fig, ax = plt.subplots(1, figsize=(10, 4), dpi=150)
            x = np.arange(len(experiments[0].manager.fitted_acr_models))
            acr_labels = [acr for acr in experiments[0].manager.fitted_acr_models]

            for i, exp in enumerate(experiments):
                f1_scores = []

                for acr in exp.manager.fitted_acr_models:
                    X_test = X_test_pt if exp.transf == 'pt' else X_test_st
                    y_test = exp.manager[acr].y_test
                    y_pred = exp.manager[acr].pl.predict(X_test)

                    f1_scores.append(f1_score(y_test, y_pred, average='binary', zero_division=0))
                ax.plot(range(0, len(exp.manager.fitted_acr_models)), f1_scores, alpha=0.8, marker='o', label=exp.id, color=plt.cm.get_cmap('tab10')(i))
                ax.axhline(y=0.8, color='purple', alpha=0.5, lw=1, linestyle='--')
                ax.set_xlabel('Assignatura')
                ax.set_ylabel('F-score')
                ax.set_title(title)
            
            ax.set_xticks(x)
            ax.set_xticklabels(acr_labels, rotation=90)

            plt.tight_layout()
            plt.legend(loc='upper right')

            plt.savefig(PLOTS_TMP_PATH / 'eval_cmp.png')
            plt.savefig(PLOTS_TMP_PATH / 'eval_cmp.pgf')
            
            if show: 
                plt.show()
            else:
                plt.close()
        else:
            fig, ax = plt.subplots(2, 1, figsize=(10, 5), dpi=150, gridspec_kw={'height_ratios': [5, 2]})
            x = np.arange(len(experiments[0].manager.fitted_acr_models))
            acr_labels = [acr for acr in experiments[0].manager.fitted_acr_models]

            for i, exp in enumerate(experiments):
                f1_scores = []

                for acr in exp.manager.fitted_acr_models:
                    X_test = X_test_pt if exp.transf == 'pt' else X_test_st
                    y_test = exp.manager[acr].y_test
                    y_pred = exp.manager[acr].pl.predict(X_test)

                    f1_scores.append(f1_score(y_test, y_pred, average='binary', zero_division=0))

                ax[0].plot(range(0, len(exp.manager.fitted_acr_models)), f1_scores, alpha=0.8, marker='o', label=exp.id, color=plt.cm.get_cmap('tab10')(i))
                ax[0].axhline(y=0.8, color='purple', alpha=0.5, lw=1, linestyle='--')
                ax[0].set_xlabel('Assignatura')
                ax[0].set_ylabel('F-score')
                ax[0].set_title(title)
            ax[0].legend(loc='upper right')
            ax[0].set_xticks(x)
            ax[0].set_xticklabels(acr_labels, rotation=90)


            fit_times = [exp.manager.fit_time for exp in experiments]
            exps_id   = [exp.id for exp in experiments]
            colors    = [plt.cm.get_cmap('tab10')(i) for i in range(len(experiments))]

            fit_times, exps_id, colors = zip(*sorted(zip(fit_times, exps_id, colors)))

            ax[1].barh(exps_id, fit_times, alpha=0.8, height=0.35,  color=colors)
            ax[1].set_xlabel('Temps d\'entrenament (s)')
            # ax[1].set_title('Fit Time')

            # ax[1].set_xticks(range(0, len(experiments)))
            # ax[1].set_xticklabels(exps_id, rotation=90)

            plt.tight_layout()
            plt.legend(loc='upper right')

            plt.savefig(PLOTS_TMP_PATH / 'eval_cmp.png')
            plt.savefig(PLOTS_TMP_PATH / 'eval_cmp.pgf')
            
            if show: 
                plt.show()
            else:
                plt.close()

