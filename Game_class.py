import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# problems
# 1 - leaf is colored



class DiGraph:
    def __init__(self, aj_dict: dict[int, list[int]] | None = None,
                 colors: dict[int, int] | None = None, full: bool = True):
        if aj_dict is None:
            aj_dict = {}
        self.aj_dict: dict[int, list[int]] = aj_dict  # v -> list[v]
        self.full: bool = full  # to aggregate edges or not
        if full:
            self.v: list[int] = list(aj_dict.keys())
            self.v_num: int = len(self.v)
            self.e: list[tuple[int, int]] = [(i, j) for i, lst in aj_dict.items() for j in lst]
        if colors is None:
            if not full:
                self.v: list[int] = list(aj_dict.keys())
                self.v_num: int = len(self.v)
            self.colors = dict(zip(self.v, self.v))
        else:
            self.colors = colors  # vertex to it's color
        self.nx: nx.DiGraph | None = None

    def mk_full(self) -> None:
        self.full = True
        self.v: list[int] = list(self.aj_dict.keys())
        self.v_num: int = len(self.v)
        self.e: list[tuple[int, int]] = [(i, j) for i, lst in self.aj_dict.items() for j in lst]

    def vertices(self) -> list[int]:
        if not self.full:
            self.mk_full()
        return self.v

    def edges(self) -> list[tuple[int, int]]:
        if not self.full:
            self.mk_full()
        return self.e

    def get_colors(self):
        return self.colors

    def get_pl_to_vertices(self) -> dict[int, list[int]]:
        res: dict[int, list[int]] = {}
        for key in set(self.colors.values()):
            res[key] = [old_key for (old_key, old_value) in self.colors.items() if old_value == key]
        return res

    def to_nx(self, name: str | None = None):
        if self.nx is None:
            if name is not None:
                g = nx.DiGraph(self.aj_dict, name=name)
            else:
                g = nx.DiGraph(self.aj_dict)
            self.nx = g
            return g
        else:
            return self.nx

    def demonstrate(self, pos=nx.spring_layout, **kwargs) -> None:
        g = self.to_nx()
        node_color = [self.colors[key] for key in sorted(self.colors.keys(), reverse=True)]
        nx.draw(g, pos(g), node_color=node_color, **kwargs)
        plt.show()

    def get_degree(self, outer=True) -> dict[int, int]:
        if outer:
            return dict(self.to_nx().out_degree(self.vertices()))


# player numeration from 0
class Game:
    def __init__(self, graph: DiGraph):
        self.graph: DiGraph = graph
        self.players: int = len(set(graph.get_colors().values()))
        self.round_table: np.array = self.init_table()

    def init_table(self) -> np.array:
        p_to_v: dict[int, list[int]] = self.graph.get_pl_to_vertices()
        degrees: dict[int, int] = self.graph.get_degree()
        shape: list[int] = [0] * self.players
        for player, vertices in p_to_v.items():
            for v in vertices:
                if degrees[v] > 0:
                    shape[player] += degrees[v]
                else:
                    p_to_v[player].remove(v)
        res = np.empty(shape)
        for index, value in np.ndenumerate(res):
            pass    # iterating though elements and for each element get
            # vertices and edges which we take, then get connected subgraph with all colors,
            # from which we take terminal if it exists, else return c
        return np.empty(shape)
