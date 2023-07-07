import sys
import argparse
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

from assig import Assig, PlaEstudis
from base import Quad
from taumat import CarregaTaules
from mypp import MyPrettyPrinter


BASE_PATH     = Path('../../data')
RAW_PATH      = BASE_PATH / 'raw'
INTERIM_PATH  = BASE_PATH / 'interim'

RAW_MAT_FILE_PATH  = RAW_PATH / 'matricules.anon.csv'
RAW_ACRO_FILE_PATH = RAW_PATH / 'acronims.csv'



@dataclass(frozen=True)
class Mat(object):
    """
    La matricula d'una assignatura i el resultat obtingut
    """
    assig  : Assig
    nota   : float
    notad  : str
    tipusn : str

    def __str__(self):
        nota = self.nota if self.notad not in ('MH', 'NP') else self.notad
        if self.tipusn:
            return f'{self.assig!s}={nota}[{self.tipusn}]'
        else:
            return f'{self.assig!s}={nota}'



@dataclass
class BlkMat(object):
    """El block de matricules que es matriculen simultaniament"""

    becat  : bool = False
    lstmat : list[Mat] = field(default_factory=list)

    def __str__(self):
        if self.becat:
            return '[Bec] ' + ', '.join(f'{m!s:18}' for m in self.lstmat)
        else:
            return '      ' + ', '.join(f'{m!s:18}' for m in self.lstmat)
        


@dataclass
class Exped(object):
    """
    L'expedient d'un estudiant
    """
    idexp : int
    notae : float
    viae  : str
    ordre : int
    anyn  : int
    mats  : dict[Quad, BlkMat]

    def __str__(self):
        r =  f'--- expedient: {self.idexp}\n'
        r += f'notae = {self.notae}\n'
        r += f'viae  = {self.viae}\n'
        r += f'ordre = {self.ordre}\n'
        r += f'anyn  = {self.anyn}\n'
        r += ''.join(f'{q!s}: {self.mats[q]!s}\n' for q in self.mats)
        return r
    
    def __len__(self):
        # NOTE: Afegit: retorna el nombre de matricules
        return len(self.mats.keys())
    
    def primer_quad(self):
        # NOTE: Afegit: retorna l'any de la primera matricula
        return min(q for q in self.mats.keys())

    def ultim_quad(self):
        # NOTE: Afegit: retorna l'any de la última matricula
        return max(q for q in self.mats.keys())

    def durada(self):
        # NOTE: Afegit: retorna la durada (TOTAL) de l'expedient
        return self.ultim_quad() - self.primer_quad()
    
    def completa(self, pe: PlaEstudis):
        # NOTE: Completa totes les assignatures no matriculdes
        
        for _q, mats in self.mats.items():

            # print(mats.lstmat)
            # print(len(mats.lstmat))


            for a_codi in pe._idx_codi.keys():
                
                if a_codi not in [mat.assig.codi for mat in mats.lstmat]:
                    # print('NO', a)
                    mats.lstmat.append(
                        Mat(
                            assig = Assig(codi=a_codi, acro=pe._idx_codi[a_codi]),
                            nota = None,
                            notad = None,
                            tipusn = None
                        )
                    )

                else:
                    # print('SI', a_codi, pe._idx_codi[a_codi])
                    pass

            # print(mats.lstmat)
            # print(len(mats.lstmat))

            # print()



class BDExped(object):
    """
    La base de dades estructurada d'expedients amb les seves matrícules.
    Es calcula a partir de la taula crua de matrícules.
    """

    def __init__(self, tm, ta):
        
        # NOTE: Afegit, Gaudem les taules
        self.tm = tm
        self.ta = ta

        # crea el pla d'estudis de la BD
        self.pe = PlaEstudis()

        for id_ass in self.ta:
            self.pe.add(Assig(id_ass, self.ta[id_ass]))

        # bd d'expedients: és un diccionari d'expedients
        self.bd = dict()
        for idexp in self.tm:
            # diccionari de matrícules per quadrimestre
            d = defaultdict(BlkMat)
            for matcrua in self.tm[idexp]:
                m = Mat(
                    assig  = self.pe.get(matcrua.assid),
                    nota   = matcrua.nota,
                    notad  = matcrua.notades,
                    tipusn = matcrua.tipusnota
                )
                q = Quad(matcrua.curs, matcrua.quad)
                d[q].becat = matcrua.becat
                d[q].lstmat.append(m)
            # crea expedient amb info comuna
            e = Exped(
                idexp = idexp,
                notae = self.tm.attr_d_exped(idexp, 'notaacc'),
                viae  = self.tm.attr_d_exped(idexp, 'viaacc'),
                ordre = self.tm.attr_d_exped(idexp, 'ordreass'),
                anyn  = self.tm.attr_d_exped(idexp, 'anynaix'),
                mats  = {k:d[k] for k in sorted(d)},
            )
            # afegeix expedient a la bd
            self.bd[idexp] = e

    def __str__(self):
        return '\n'.join(f'{self.bd[e]!s}' for e in self.bd)
    
    def completa(self):
        for idexp in tqdm(self.bd, desc="Extensió dels expedients"):
            self.bd[idexp].completa(self.pe)

    def edat_classificador(self, edat) -> int:
        if edat >= 18 and edat <= 20:
            # Primer any d'universitat
            return 0
        
        elif edat >= 21 and edat <= 24:
            # Recent pero que s'incorpora "tard" / via FP etc.
            return 1
        
        else:
            # Estudiant que ha seguit una via no "tradicional" / "different"
            return 2


    def _get_df_by_idexp(self, idexp):
        MAXQ = 10
        est = self.bd[idexp]
        
        # dades fixes
        df = pd.DataFrame() # NOTE: Added

        try:
            cols = {}

            # Informació comuna
            cols['EXPID'] = est.idexp
            cols['EDAT'] = self.edat_classificador(est.primer_quad().any() - est.anyn)
            cols['VIA']  = est.viae
            cols['ORDRE']= est.ordre
            cols['NACC'] = est.notae

            # Dades d'expedient decalades
            df_lst = []
            for i in range(min(MAXQ, est.durada())):
                # Si est.durada es > MAXQ iterarem nomes fins MAXQ-1
                # emplenem quadrimestres q=[0,i]
                for q in list(est.mats.keys())[:i+1]:
                    expq = est.mats[q]
                    
                    becat = est.mats[q].becat
                    
                    cols[f'{i}:becat'] = becat
                    # Emplenem les columnes assignatures
                    for mat in expq.lstmat:
                        nomc = '{}:{}'.format(i, self.pe._idx_codi[mat.assig.codi].acro)
                        # print(nomc)
                        # print(nomc)
                        

                        if mat.nota is None:
                            cols[nomc+'.n'] = 0.0
                            cols[nomc+'.m'] = False
                        else:
                            cols[nomc+'.n'] = mat.nota
                            cols[nomc+'.m'] = True

                df_lst.append(pd.DataFrame(cols, index=[0], copy=True))
                df = pd.concat(df_lst, ignore_index=True)

        except Exception as e:
            print(e)
            raise

        return df
    

    def get_all_df(self):
        df = pd.DataFrame()

        for idexp in tqdm(self.bd, desc='Construïnt el dataset (base)'):
            sys.stdout.write('\r')
            sys.stdout.flush()
            
            tmp_df  = self._get_df_by_idexp(idexp)
            df = pd.concat([df, tmp_df], ignore_index=True)
            print('.', end='', flush=True)

        return df
    



def parse_args():
    parser = argparse.ArgumentParser(description='Exporta un nou fixer *.csv del Dataset (base).')
    parser.add_argument('--nomf', type=str, help='El nom del fixer resultant')
    args = parser.parse_args()

    # Comprovem que no existeixi un fitxer amb el mateix nom (seguretat/prevenció)
    file_path = INTERIM_PATH / args.nomf
    if file_path.exists():
        raise FileExistsError(f'El fitxer {args.nomf} ja existeix')

    return file_path


if __name__ == '__main__':

    try:
        
        file_path = parse_args()

        pp = MyPrettyPrinter(indent=2)

        (tm, ta) = CarregaTaules(
                nom_mat=RAW_MAT_FILE_PATH,
                nom_acr=RAW_ACRO_FILE_PATH,
                reporta=False
            )
        
        bd = BDExped(tm, ta)
        bd.completa()
        
        df = bd.get_all_df()

        df.to_csv(INTERIM_PATH / file_path, index=False)
    
    except Exception as e:
        print(e)