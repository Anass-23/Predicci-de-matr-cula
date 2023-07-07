from dataclasses import dataclass
from typing import Union


@dataclass(order=True, frozen=True)
class Assig(object):
    """
    Una assignatura
    """
    # codi : str
    # acro : int

    codi : int
    acro : str
    nom  : str = None

    def __str__(self):
        return f'{self.acro}({self.codi})'



class PlaEstudis(object):
    """
    El conjunt d'assignatures d'un pla d'estudis.
    """
    def __init__(self):
        self._idx_codi = dict()
        self._idx_acro = dict()

    def add(self, ass: Assig) -> None:
        self._idx_codi[ass.codi] = ass
        self._idx_acro[ass.acro] = ass

    def get(self, idx: Union[str, int]) -> Assig:
        if type(idx) is str:
            return self._idx_acro[idx]
        else:
            return self._idx_codi[idx]
