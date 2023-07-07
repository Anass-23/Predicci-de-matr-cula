from functools import total_ordering

@total_ordering
class Quad(object):
    """
    Un quadrimestre en tant que data en el temps
    Quad(2021,1) = primer quadrimestre del curs 2021  (i.e. quad tardor)
    """

    def __init__(self, a, q):
        self.q = (a, q-1)

    def __str__(self):
        return '{}·{}'.format(self.q[0], self.q[1]+1 )

    def __repr__(self):
        return 'Quad({},{})'.format(self.q[0], self.q[1]+1 )

    def __eq__(self, q):
        return self.q == q.q

    def __lt__(self, q):
        return self.q < q.q

    def __hash__(self):
        return hash(self.q)

    def __sub__(self, q):
        return 2*(self.q[0] - q.q[0]) + self.q[1] - q.q[1]

    def __int__(self):
        return 2*self.q[0] + self.q[1]

    def __index__(self):
        return int(self)

    def __add__(self, op):
        a = self.q[0] + op // 2 + (self.q[1] + op % 2) // 2
        b = (self.q[1] + op % 2) % 2
        return Quad(a,b+1)

    def any(self):
        return self.q[0]
