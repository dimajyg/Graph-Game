import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product, chain
from collections import defaultdict as dd
from boltons.setutils import IndexedSet
# problems
# 1 - leaf is colored
# 2 - players enumeration only from 0


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
            self.colors = {v: c for v, c in colors.items() if v in aj_dict.keys()}  # vertex to it's color
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

    def get_connects(self, v: int) -> list[int]:
        return sorted(self.aj_dict[v])

    def get_terminals(self) -> list[int]:
        res: list[int] = []
        for v, neighbs in self.aj_dict.items():
            if len(neighbs) == 0:
                res.append(v)
        return res

    def get_pl_to_vertices(self) -> dict[int, list[int]]:   # player to his's vertices
        res: dict[int, list[int]] = {}
        terminals: set[int] = set(self.get_terminals())
        for key in set(self.colors.values()):
            if len(y := sorted([old_key for (old_key, old_value) in self.colors.items()
                                if old_value == key and old_key not in terminals])) > 0:
                res[key] = y
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
        node_color = [self.colors[key] for key in sorted(self.colors.keys())]
        nx.draw(g, pos(g), node_color=node_color, **kwargs)
        plt.show()

    def get_degree(self, outer=True) -> dict[int, int]:
        if outer:
            return dict(self.to_nx().out_degree(self.vertices()))

    def get_leaf(self, edges):
        pass

    def dfs_util(self, v, visited):
        visited.add(v)
        print(v, end=" ")
        # recur for all the vertices adjacent to this vertex
        for neighbour in self.aj_dict[v]:
            if neighbour not in visited:
                self.dfs_util(neighbour, visited)
        # The function to do DFS traversal. It uses recursive DFSUtil

    def dfs(self):
        # create a set to store all visited vertices
        visited = set()
        # call the recursive helper function to print DFS traversal starting from all
        # vertices one by one
        for vertex in self.vertices():
            if vertex not in visited:
                self.dfs_util(vertex, visited)

    def _ffr_util(self, v, visited):    # find first end of game (root)
        visited.add(v)
        # recur for all the vertices adjacent to this vertex
        for neighbour in self.aj_dict[v]:
            if neighbour not in visited:
                self._ffr_util(neighbour, visited)
        # The function to do DFS traversal. It uses recursive DFSUtil
        return visited

    def ffr(self):
        # create a set to store all visited vertices
        visited = IndexedSet()
        # call the recursive helper function to print DFS traversal starting from all
        # vertices one by one
        for vertex in self.vertices():
            if vertex not in visited:
                tmp: IndexedSet[int] | None = self._ffr_util(vertex, visited)
                if tmp is not None:
                    return tmp


def list_edges_to_digraph(edges: list[tuple[int, int]], colors: dict[int, int]) -> DiGraph:
    result = {}
    for i in edges:
        result.setdefault(i[0], []).append(i[1])
    for i in chain(*result.values()):
        if i not in result.keys():
            result[i] = []
    return DiGraph(result, colors)


# player numeration from 0
class Game:
    def __init__(self, graph: DiGraph, v0: int):
        self.graph: DiGraph = graph
        self.v0: int = v0
        self.players: list[int] = sorted(graph.get_pl_to_vertices().keys())
        self.round_table: np.array = self.init_table()

    def init_table(self) -> np.array:
        p_to_v: dict[int, list[int]] = self.graph.get_pl_to_vertices()
        degrees: dict[int, int] = self.graph.get_degree()
        shape: list[int] = [0] * len(self.players)
        variants: dd[int, dict[int, list[int]]] = dd(dict)    # player:vertex:connections
        #
        tmp: dd[int, list[list[tuple]]] = dd(list)
        #
        for player, vertices in p_to_v.items():
            for v in vertices:
                if degrees[v] > 0:
                    shape[player] = shape[player]*degrees[v] if shape[player] else degrees[v]
                    variants[player][v] = self.graph.get_connects(v)
                    tmp[player].append(list(product([v], variants[player][v])))
                else:
                    p_to_v[player].remove(v)
        #   maybe should be changed
        v_simple: dict[int, list[tuple[tuple, ...]]] = {k: list(product(*v)) for k, v in tmp.items()}
        #   player:(v,v->neig)
        #
        print(v_simple)
        res = np.empty(shape)
        for indexes, value in np.ndenumerate(res):
            # print(indexes, value)
            edges: list[tuple[int, int]] = []
            for player, numer in enumerate(indexes):
                edges += v_simple[player][numer]
            subgraph: DiGraph = list_edges_to_digraph(edges, self.graph.get_colors())
            # subgraph.demonstrate()
            # print(edges)
            path = subgraph.ffr()
            if len(self.graph.aj_dict[path[-1]]) == 0:
                res[indexes] = path[-1]
            else:
                res[indexes] = -1   # --------------------------------------- тут менять если решим разделять циклы
            # print(subgraph.ffr())
            # print("")
            # if self.v0 not in subgraph.vertices():
            #     pass
            pass    # iterating though elements and for each element get
            # vertices and edges which we take, then get connected subgraph with all colors,
            # from which we take terminal if it exists, else return c
        print(res)
        return res
# a = DiGraph({1:[2,4],2:[3,5],3:[],4:[5,6],5:[4,7],6:[],7:[]},{1:0,2:1,3:3,4:2,5:1,6:6,7:7})
#
# Game(a, 1)