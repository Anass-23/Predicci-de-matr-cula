import pandas as pd
from tqdm import tqdm
from typing import *
from pathlib import Path
# from taumat import CarregaTaules
from data.taumat import CarregaTaules



BASE_PATH      = Path('../../data')
RAW_PATH       = BASE_PATH / 'raw'
INTERIM_PATH   = BASE_PATH / 'interim'
PROCESSED_PATH = BASE_PATH / 'processed'

RAW_MAT_FILE_PATH  = RAW_PATH / 'matricules.anon.csv'
RAW_ACRO_FILE_PATH = RAW_PATH / 'acronims.csv'



class Dataset(object):
    def __init__(self, file: str = None) -> None:
        self.raw_df =  pd.read_csv(file, low_memory=False) if file else None
        self.ta      = None 
        self.dataset = None
    
    def get_raw_df(self) -> pd.DataFrame:
        """Retorna el conjunt de dades crues preprocessades

        Returns:
            pd.DataFrame: El conjunt de dades crues preprocessades
        """
        return self.raw_df

    def add_ta(self, ta) -> None:
        """Afegeix la taula `ta` d'acrònims

        Args:
            ta (TaulaAcronims): Taula d'acrònims
        """
        self.ta = ta

    def transform(self) -> pd.DataFrame:
        """Aplica la transformació de dades per crear i retornar un dataset

        Raises:
            NotImplemented: Ha de ser implementada per una subclasse de `Dataset`

        Returns:
            pd.DataFrame: Les dades crues transformades
        """
        raise NotImplemented

    def load_data(self) -> Tuple[float, str]:
        """Carrega les dades d'entrenament

        Raises:
            NotImplemented: Ha de ser implementada per una subclasse de `Dataset`

        Returns:
            Tuple[float, str]: Dades de la forma (X, y)
        """
        raise NotImplemented



class PrimeraTransformacio(Dataset):
    def __init__(self, file: str = None) -> None:
        super().__init__(file)

    def transform(self) -> pd.DataFrame:
        # Dades "crues"
        raw_df = self.get_raw_df()
        # raw_df = raw_df.drop(
        #     # columns=['EDAT', 'VIA', 'ORDRE', 'NACC']
        #     columns=['EDAT']
        # )

        # Primerament copiem les dades crues
        self.dataset = raw_df.copy(deep=True)
        # print(raw_df)

        num_acr = len(self.ta.get_acrlst())
        # print(num_acr)
        for acr in self.ta.get_acrlst():
            if acr:
                self.dataset[acr] = None

        indx = 0
        indexes_to_delete = []
        # for expid in raw_df['EXPID'].unique()[:2]:
        # for expid in raw_df['EXPID'].unique():
        for expid in tqdm(raw_df['EXPID'].unique(), desc='Contruïnt dataset (PrimeraTransformacio)'):
            # print(indx)
            num_mat, _ = raw_df[raw_df['EXPID'] == expid].shape
            # print(f'[LOADING] {expid}')
            # print(f'[INFO]    {num_mat} matricules')

            if num_mat == 1:
                # print(f'[DELETED] Reason: 1 matrícula')
                # self.dataset = self.dataset.drop(indx)
                indexes_to_delete.append(indx)
                indx += 1

            
            else:
                for mat in range(0, num_mat):
                    # print(mat, num_mat-1)
                    if mat == 0:
                        r = raw_df[raw_df['EXPID'] == expid].loc[indx:indx, raw_df.columns.str.startswith(f'{mat}:')]
                        next_assigs = [acr.split(':')[1].strip('.m') for acr, v in r.loc[indx, r.columns.str.endswith(f'.m')].to_dict().items() if v]
                        # print(f'[MATR]    {expid} -> (Mat {mat}): {next_assigs}\n')

                    elif mat >= 0:
                        r = raw_df[raw_df['EXPID'] == expid].loc[indx:indx, raw_df.columns.str.startswith(f'{mat}:')]
                        next_assigs = [acr.split(':')[1].strip('.m') for acr, v in r.loc[indx, r.columns.str.endswith(f'.m')].to_dict().items() if v]
                        # print(f'[MATR]    {expid} -> (Mat {mat}): {next_assigs}')

                        mapping = self.dataset.iloc[indx-1, -num_acr:].to_dict()
                        # try:
                        # except:
                        #     print(1774, '->', indx)
                        #     print(self.dataset.shape)
                        #     print(self.dataset)
                        #     return

                        for k in mapping.keys(): mapping[k] = False
                        for assig in next_assigs: mapping[assig] = True
                        # print(f'[NEXT]    Mat({mat-1}) -> {next_assigs}\n')
                        
                        self.dataset.loc[indx-1, self.dataset.columns[-num_acr:]] = mapping.values()
                    
                        if mat == num_mat-1:
                            r = raw_df[raw_df['EXPID'] == expid].loc[indx:indx, raw_df.columns.str.startswith(f'{mat}:')]
                            # print(f'[MATR]    {expid} -> (Mat {mat}): {next_assigs}')

                            # self.dataset = self.dataset.drop(indx) # Eliminem del dataset la última matrícula (no hi ha matrícula següent)
                            indexes_to_delete.append(indx)

                            # print(f'[DELETED] {expid} -> Útima matricula')
                    indx += 1

        # print(f'[INFO]    Finished', '\n')

        # print(indexes_to_delete)
        # print(len(indexes_to_delete))
        # print(self.dataset.shape)
        # print(self.dataset.shape)

        # # self.dataset.to_csv('primerDataset_all.csv', index=True)

        self.dataset = self.dataset.drop(indexes_to_delete)
        # # self.dataset.to_csv('primerDataset-informacio.csv', index=True)
        return self.dataset

    # def transform(self) -> pd.DataFrame:
    #     # Dades "crues"
    #     raw_df = self.get_raw_df()
    #     # raw_df = raw_df.drop(
    #     #     # columns=['EDAT', 'VIA', 'ORDRE', 'NACC']
    #     #     columns=['EDAT']
    #     # )

    #     # Primerament copiem les dades crues
    #     self.dataset = raw_df.copy(deep=True
    #                                )
    #     num_acr = len(self.ta.get_acrlst()) - 1
    #     # print(self.ta.get_acrlst())
    #     for acr in self.ta.get_acrlst():
    #         if acr:
    #             self.dataset[acr] = None
    
    #     # print(self.dataset.columns[-num_acr:], '\n')
    
    #     # Transformació
    #     indx = 0
    #     indexes_to_delete = []
    #     # for expid in raw_df['EXPID'].unique()[:2]:
    #     for expid in raw_df['EXPID'].unique():
    #         # print(indx)
    #         num_mat, _ = raw_df[raw_df['EXPID'] == expid].shape
    #         # print(f'[LOADING] {expid}')
    #         # print(f'[INFO]    {num_mat} matricules')

    #         if num_mat == 1:
    #             # print(f'[DELETED] Reason: 1 matrícula')
    #             # self.dataset = self.dataset.drop(indx)
    #             indexes_to_delete.append(indx)
    #             indx += 1

    #         else:
    #             for mat in range(0, num_mat):
    #                 # print(mat, num_mat-1)
    #                 if mat == 0:
    #                     r = raw_df[raw_df['EXPID'] == expid].loc[indx:indx, raw_df.columns.str.startswith(f'{mat}:')]
    #                     next_assigs = [acr.split(':')[1].strip('.m') for acr, v in r.loc[indx, r.columns.str.endswith(f'.m')].to_dict().items() if v]
    #                     # print(f'[MATR]    {expid} -> (Mat {mat}): {next_assigs}\n')

    #                 elif mat >= 0:
    #                     r = raw_df[raw_df['EXPID'] == expid].loc[indx:indx, raw_df.columns.str.startswith(f'{mat}:')]
    #                     next_assigs = [acr.split(':')[1].strip('.m') for acr, v in r.loc[indx, r.columns.str.endswith(f'.m')].to_dict().items() if v]
    #                     # print(f'[MATR]    {expid} -> (Mat {mat}): {next_assigs}')

    #                     mapping = self.dataset.iloc[indx-1, -num_acr:].to_dict()
    #                     # try:
    #                     # except:
    #                     #     print(1774, '->', indx)
    #                     #     print(self.dataset.shape)
    #                     #     print(self.dataset)
    #                     #     return

    #                     for k in mapping.keys(): mapping[k] = False
    #                     for assig in next_assigs: mapping[assig] = True
    #                     # print(f'[NEXT]    Mat({mat-1}) -> {next_assigs}\n')
                        
    #                     self.dataset.loc[indx-1, self.dataset.columns[-num_acr:]] = mapping.values()
                    
    #                     if mat == num_mat-1:
    #                         r = raw_df[raw_df['EXPID'] == expid].loc[indx:indx, raw_df.columns.str.startswith(f'{mat}:')]
    #                         # print(f'[MATR]    {expid} -> (Mat {mat}): {next_assigs}')

    #                         # self.dataset = self.dataset.drop(indx) # Eliminem del dataset la última matrícula (no hi ha matrícula següent)
    #                         indexes_to_delete.append(indx)

    #                         # print(f'[DELETED] {expid} -> Útima matricula')
    #                 indx += 1

    #     print(f'[INFO]    Finished', '\n')

    #     print(indexes_to_delete)
    #     print(len(indexes_to_delete))
    #     print(self.dataset.shape)
    #     print(self.dataset.shape)

    #     # self.dataset.to_csv('primerDataset_all.csv', index=True)

    #     self.dataset = self.dataset.drop(indexes_to_delete)
    #     # self.dataset.to_csv('primerDataset-informacio.csv', index=True)
    #     return self.dataset

    def load_data(self, file: str) -> Tuple[float, str]:
        # return super().load_data()
        dataset = pd.read_csv(file, low_memory=False, index_col=0)
        num_acr = len(self.ta.get_acrlst())


        # X
        X = dataset.iloc[:, : -num_acr]
        # X[X.columns[X.columns.str.endswith('.m') & X.isna().all()]] = X[X.columns[X.columns.str.endswith('.m') & X.isna().all()]].fillna(False)
        # X[X.columns[X.columns.str.endswith('.n') & X.isna().all()]] = X[X.columns[X.columns.str.endswith('.n') & X.isna().all()]].fillna(0.0)

        X[X.columns[X.columns.str.endswith('.m')]]    = X[X.columns[X.columns.str.endswith('.m')]].fillna(False)
        X[X.columns[X.columns.str.endswith('.n')]]    = X[X.columns[X.columns.str.endswith('.n')]].fillna(0.0)
        X[X.columns[X.columns.str.endswith('becat')]] = X[X.columns[X.columns.str.endswith('becat')]].fillna(False)
        X[['VIA', 'ORDRE', 'NACC']]                   = X[['VIA', 'ORDRE', 'NACC']].fillna(0)


        # data.loc[:, data.columns.str.endswith('becat')] = data.loc[:, data.columns.str.endswith('becat')].fillna(False)

        # cols_with_becat = data.columns[data.columns.str.endswith('becat')]

        # data[cols_with_becat] = data[cols_with_becat].fillna(False)


        # y
        y = dataset.iloc[:, -num_acr:]

        return (X, y)



class SegonaTransformacio(Dataset):
    def __init__(self, file: str = None) -> None:
        super().__init__(file)

    def transform(self) -> pd.DataFrame:
        # Dades "crues"
        raw_df = self.get_raw_df()

        columns = ['EXPID', 'EDAT', 'VIA', 'ORDRE', 'NACC', 'BECAT']
        num_acr = len(self.ta.get_acrlst())

        for acr in self.ta.get_acrlst():
            if acr:
                columns.append(f'{acr}.da')
                columns.append(f'{acr}.n')
                columns.append(f'{acr}.m')
        
        for acr in self.ta.get_acrlst():
            if acr:
                columns.append(f'{acr}')
        
        self.dataset = pd.DataFrame(
            columns=columns,
            data=[]
        )
        
        for col in columns[:5]:
            self.dataset[col] = self.raw_df[col]        
        
        indx              = 0
        indexes_to_delete = []
        for expid in tqdm(raw_df['EXPID'].unique()[:], desc='Contruïnt dataset (SegonaTransformacio)'):
        
            num_mat, _ = raw_df[raw_df['EXPID'] == expid].shape
            assig_hist = {}

            if num_mat == 1:
                # print(f'[DELETED] Reason: 1 matrícula')
                indexes_to_delete.append(indx)
                indx += 1

            else:
                for mat in range(0, num_mat):
                    # print(mat, num_mat-1)
                    if mat == 0:
                        r = raw_df[raw_df['EXPID'] == expid].loc[indx:indx, raw_df.columns.str.startswith(f'{mat}:')]
                        next_assigs = [acr.split(':')[1].strip('.m') for acr, v in r.loc[indx, r.columns.str.endswith(f'.m')].to_dict().items() if v]
                        notes_assigs = [(acr.split(':')[1].strip('.n'), v) for acr, v in r.loc[indx, r.columns.str.endswith(f'.n')].to_dict().items() if v]
                        # print('becat', r.at[indx, f'{mat}:becat'])

                        self.dataset.at[indx, 'BECAT'] = r.at[indx, f'{mat}:becat']
                        for acr, v in notes_assigs:
                            if acr in assig_hist:
                                assig_hist[acr].append(v)
                            else:
                                assig_hist[acr] = [v]
                            self.dataset.at[indx, f'{acr}.da'] = len(assig_hist[acr])
                            self.dataset.at[indx, f'{acr}.n']  = v
                            self.dataset.at[indx, f'{acr}.m']  = True

                        # print(acr, v, assig_hist)
                        self.dataset
                        # print(f'[MATR]    {expid} -> (Mat {mat}): {next_assigs}\n')
                        # print(f'[MATR]    {expid} -> (Mat {mat}): {notes_assigs}\n')


                    elif mat > 0:
                        r = raw_df[raw_df['EXPID'] == expid].loc[indx:indx, raw_df.columns.str.startswith(f'{mat}:')]
                        next_assigs = [acr.split(':')[1].strip('.m') for acr, v in r.loc[indx, r.columns.str.endswith(f'.m')].to_dict().items() if v]
                        notes_assigs = [(acr.split(':')[1].strip('.n'), v) for acr, v in r.loc[indx, r.columns.str.endswith(f'.n')].to_dict().items() if v]
                        self.dataset.at[indx, 'BECAT'] = r.at[indx, f'{mat}:becat']
                        for acr, v in notes_assigs:
                            if acr in assig_hist:
                                assig_hist[acr].append(v)
                            else:
                                assig_hist[acr] = [v]
                            self.dataset.at[indx, f'{acr}.da'] = len(assig_hist[acr])
                            self.dataset.at[indx, f'{acr}.n']  = v
                            self.dataset.at[indx, f'{acr}.m']  = True

                        # print(f'[MATR]    {expid} -> (Mat {mat}): {next_assigs}')

                        mapping = self.dataset.iloc[indx-1, -num_acr:].to_dict()

                        for k in mapping.keys(): mapping[k] = False
                        for assig in next_assigs: mapping[assig] = True
                        # print(f'[NEXT]    Mat({mat-1}) -> {next_assigs}\n')
                        
                        self.dataset.loc[indx-1, self.dataset.columns[-num_acr:]] = mapping.values()
                    
                        if mat == num_mat-1:
                            r = raw_df[raw_df['EXPID'] == expid].loc[indx:indx, raw_df.columns.str.startswith(f'{mat}:')]
                            # print(f'[MATR]    {expid} -> (Mat {mat}): {next_assigs}')

                            # self.dataset = self.dataset.drop(indx) # Eliminem del dataset la última matrícula (no hi ha matrícula següent)
                            indexes_to_delete.append(indx)

                            # print(f'[DELETED] {expid} -> Útima matricula')
                    indx += 1

        # print(f'[INFO]    Finished', '\n')
        # print(indexes_to_delete)
        # print(len(indexes_to_delete))
        # print(self.dataset.shape)

        # Delete the expid's (num_mat == 1)
        self.dataset = self.dataset.drop(indexes_to_delete)

        # Empty mat boolean (na to False)
        # NOTE: Before ffill!!!
        empty_mat_na = [col for col in self.dataset.columns if col.endswith('.m')]
        self.dataset.loc[:, empty_mat_na] = self.dataset.loc[:, empty_mat_na].fillna(False)

        # Forward fill (students history)
        self.dataset = self.dataset.fillna(method='ffill')

        # Assigs not mat (darrer any, strategy -> da=0)
        assig_not_mat = [col for col in self.dataset.columns if col.endswith('.da')]
        self.dataset.loc[:, assig_not_mat] = self.dataset.loc[:, assig_not_mat].fillna(0)

        # Assigs not mat (quadris més enllà ..., nota -> n=0)
        assig_not_mat = [col for col in self.dataset.columns if col.endswith('.n')]
        self.dataset.loc[:, assig_not_mat] = self.dataset.loc[:, assig_not_mat].fillna(0.0)


   
        return self.dataset

    def load_data(self, file: str) -> Tuple[float, str]:
        dataset = pd.read_csv(file, low_memory=False, index_col=0)
        num_acr = len(self.ta.get_acrlst())

        X = dataset.iloc[:, : -num_acr]
        X[X.columns[X.columns.str.endswith('.m')]]    = X[X.columns[X.columns.str.endswith('.m')]].fillna(False)
        X[X.columns[X.columns.str.endswith('.n')]]    = X[X.columns[X.columns.str.endswith('.n')]].fillna(0.0)
        X[X.columns[X.columns.str.endswith('becat')]] = X[X.columns[X.columns.str.endswith('becat')]].fillna(False)
        X[['VIA', 'ORDRE', 'NACC']]                   = X[['VIA', 'ORDRE', 'NACC']].fillna(0)

        y = dataset.iloc[:, -num_acr:]

        return (X, y)

if __name__ == '__main__':
    # Taula acrònims
    TRANSF = 2
    (tm, ta) = CarregaTaules(
        nom_mat=RAW_MAT_FILE_PATH,
        nom_acr=RAW_ACRO_FILE_PATH,
        reporta=False
    )

    # Càrrega de les dades crues per a la 1a transformació
    pt = None
    if TRANSF == 1:
        pt = PrimeraTransformacio(file=INTERIM_PATH / 'dataset_base.csv')
    else:
        pt = SegonaTransformacio(file=INTERIM_PATH / 'dataset_base.csv')
    
    pt.add_ta(ta)

    # Transformació
    ds = pt.transform() 

    # Emmagatzemament
    ds.to_csv(PROCESSED_PATH / 'segonDataset.csv', index=False)