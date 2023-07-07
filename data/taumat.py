import csv
from collections import namedtuple, defaultdict
from data.mypp import MyPrettyPrinter
# from mypp import MyPrettyPrinter
from tqdm import tqdm

pp = MyPrettyPrinter(indent=2)

class TaulaAcronims(object):
    """
    La taula d'acronims de les assignatures
    """

    def __init__(self, id_assignatures, nomf):
        """
        Inicialitza la taula del fitxer `nomf` i n'obté només les
        assignatures d'identificador a `id_assignatures`.
        """
        with open(nomf, newline='') as fitxer:
            iterador = csv.reader(fitxer)
            next(iterador) # salta capçalera
            self.acro = {}
            for row in tqdm(iterador, desc='Carregant la taula acronims', miniters=0, mininterval=0):
                if int(row[0]) in id_assignatures:
                    self.acro[int(row[0])] = row[1]

        self.iacro = {self.acro[k]:k for k in self.acro}
        if len(self.iacro) != len(self.acro):
            raise Exception('Acronims repetits')
        

        # with open(nomf, newline='') as fitxer:
        #     iterador = csv.reader(fitxer)
        #     next(iterador) # salta capçalera
        #     self.acro = {int(row[0]):row[1] for row in iterador
        #                  if int(row[0]) in id_assignatures}
        # self.iacro = {self.acro[k]:k for k in self.acro}
        # if len(self.iacro) != len(self.acro):
        #     raise Exception('Acronims repetits')

    def __getitem__(self,key):
        return self.acro[key]

    def _codis_sense_acronim(self, id_assignatures):
        """Llista dels codis necessaris sense acronim conegut"""
        return [c for c in id_assignatures if c not in self.acro]

    def get_assid(self, acro):
        return self.iacro[acro]

    def get_acrlst(self):
        return list(self.iacro.keys())

    def __iter__(self):
        for id_ass in self.acro.keys():
            yield id_ass

    def get_assidlst(self):
        return list(self.acro.keys())


class TaulaMatricules(object):
    """La taula de dades de matrícula tal i com arriba de Prisma amb
    alguns petits filtres bàsics. Indexada i agrupada per numero
    d'expedient.

    Estructura: dict[numexp, list[MatriculaCrua]]
    """

    def __init__(self, fitxer_matricules):
        self.expedients  = defaultdict(list)
        self.cjt_codiass = set()
        self._llegeix_matricules(fitxer_matricules)
        self._elimina_repetides()

    def _camps(self):
        return (
            'expid',
            'curs',
            'quad',
            'assid',
            'nota',
            'notades',
            'tipusnota',
            'becat',
            'anynaix',
            'viaacc',
            'ordreass',
            'notaacc',
        )

    def _tipifica_fila(self, r):
        """
        Les dades llegides són sempre strings i aquest mètode el
        converteix al tipus més escaient.
        """
        # r[0] = int(r[0])
        r[0] = str(r[0]) # NOTE: Changed to str
        r[1] = int(r[1])
        r[2] = int(r[2])
        r[3] = int(r[3])
        r[4] = float(r[4]) if r[4] else None
        r[6] = r[6] if r[6] else None
        r[7] = r[7] == 'SI'
        r[8] = int(r[8]) if r[8] else None
        r[9] = int(r[9]) if r[9] else None
        r[10] = int(r[10]) if r[10] else None
        r[11] = float(r[11]) if r[11] else None

    def _llegeix_matricules(self, nomf):
        MatriculaCrua = namedtuple('MatriculaCrua', self._camps(), rename=True)
        with open(nomf, newline='') as fitxer:
            iterador = csv.reader(fitxer)
            next(iterador)
            # for fila in iterador:
            for fila in tqdm(iterador, desc="Carregant la taula matricules", miniters=0, mininterval=0):
                self._tipifica_fila(fila)
                r = MatriculaCrua(*fila)
                self.expedients[r.expid].append(r)
                self.cjt_codiass.add(r.assid)

    def _elimina_repetides(self):
        """Elimina matrícules repetides en dades origen"""
        self.expedients = {
            expid: list(set(self.expedients[expid]))
            for expid in self.expedients
        }

    def _esborra_codis_assig(self, codis):
        """Esborra les matrícules d'assignatures amb el codi a `codis`"""
        for id_exped in self.expedients:
            self.expedients[id_exped][:] = [m for m in self.expedients[id_exped]
                                            if m.assid not in codis]
            self.cjt_codiass -= set(codis)

    def attr_d_exped(self, id_exped, attr):
        """Atribut `atr` d'un expedient `id_exped`"""
        return getattr(self.expedients[id_exped][0],attr)

    def cjt_ids_assig(self):
        # get the set of subjects
        return frozenset(self.cjt_codiass)

    def cjt_ids_exped(self):
        """Conjunt d'identificadors d'expedients"""
        return frozenset(self.expedients.keys())

    def __getitem__(self, id_exped):
        """Accedeix a un expedient per identificador"""
        return self.expedients[id_exped]

    def __str__(self):
        pp.pprint(self.expedients)

    def __iter__(self):
        for id_exped in self.expedients:
            yield id_exped


def CarregaTaules(nom_mat, nom_acr, reporta=False):
    """Carrega les taules crues dels fitxers i retorna (tm, ta)"""
    tm = TaulaMatricules(nom_mat)
    ta = TaulaAcronims(tm.cjt_ids_assig(), nom_acr)
    codis_no_resolts = ta._codis_sense_acronim(tm.cjt_ids_assig())
    if reporta:
        pp.pprint(ta.acro)
        print('Codis desconeguts: ', codis_no_resolts)

    tm._esborra_codis_assig(codis_no_resolts)
    return (tm, ta)
