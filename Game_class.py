import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product, chain, permutations
from collections import defaultdict as dd
from boltons.setutils import IndexedSet
from pysat.formula import CNF
from pysat.solvers import Solver
from useful_funcs import filter_equal
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
        if v not in self.aj_dict.keys():
            return visited
        for neighbour in self.aj_dict[v]:
            if neighbour not in visited:
                self._ffr_util(neighbour, visited)
        # The function to do DFS traversal. It uses recursive DFSUtil
        return visited

    def ffr(self, start):
        # create a set to store all visited vertices
        visited = IndexedSet()
        tmp: IndexedSet[int] | None = self._ffr_util(start, visited)
        if tmp is not None:
            return tmp
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
        self.cnf = None
        self.compares_codes : dict[tuple[int,int,int], int] = None
        self.payoffs : dict[int,list[int]] = None

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
        res = np.empty(shape)
        for indexes, value in np.ndenumerate(res):
            # print(indexes, value)
            edges: list[tuple[int, int]] = []
            for player, numer in enumerate(indexes):
                edges += v_simple[player][numer]
            subgraph: DiGraph = list_edges_to_digraph(edges, self.graph.get_colors())
            # subgraph.demonstrate()
            # print(edges)
            path = subgraph.ffr(self.v0)
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
        return res

    def make_CNF(self) -> CNF:
        cnf = CNF(from_clauses=[])
        self.compares_codes = {}
        encode_index = 1

        def add_encode_and_pass_code(player: int, value_1: int, value_2: int):
            nonlocal encode_index
            if (player, min(value_1, value_2), max(value_1,value_2)) not in self.compares_codes:
                self.compares_codes[(player, min(value_1, value_2), max(value_1,value_2))] = encode_index
                encode_index += 1
            return self.compares_codes[(player, min(value_1, value_2), max(value_1,value_2))] if value_1 < value_2 else -self.compares_codes[(player, min(value_1, value_2), max(value_1,value_2))]
        
        for coords, value in np.ndenumerate(self.round_table):
            # print("!!!!", coords, value)
            clause = np.array([])
            str_coords_2 = list(map(lambda x: str(x) + ":" + str(x+1), coords))
            for player, axis in enumerate(coords):
                if player == 0:
                    cmd_str = "self.round_table[:," + ",".join(str_coords_2[1:])+"]"
                elif player == len(coords) - 1:
                    cmd_str = "self.round_table[" + ",".join(str_coords_2[:-1]) + ",:" + "]"
                else:
                    cmd_str = "self.round_table["+",".join(str_coords_2[:player]) + ",:," + ",".join(str_coords_2[player+1:])+"]"
                # print(cmd_str)
                # print(eval(cmd_str))
                clause = np.concatenate([clause, np.vectorize(lambda x: add_encode_and_pass_code(player,x,value), otypes=[int])(filter_equal(np.unique(np.array(eval(cmd_str)).flatten()),value))])
            cnf.append(clause.astype(int).tolist())
            # print(self.compares_codes)
            # print(value, clause)
        outcomes = np.unique(self.round_table)
        for player in range(len(self.round_table.shape)):
            for L in range(2,len(outcomes) + 1):
                for loop in permutations(outcomes, L):
                    clause = np.array(list(map(lambda x: add_encode_and_pass_code(player,x[0],x[1]), list(zip(loop,loop[1:])) + [(loop[-1], loop[0])])))
                    # print(clause)
                    # print(self.compares_codes, loop)
                    cnf.append(clause.tolist())
        # print(cnf.clauses)
        self.cnf = cnf
        return self.cnf
    
    def get_compares(self):
        self.payoffs = {}
        code_to_pair = {v: k for k, v in self.compares_codes.items()}
        with Solver(bootstrap_with=self.cnf.clauses) as solver:
            if solver.solve():
                for compare in solver.get_model():
                    if compare > 0:
                        player, better, worse = code_to_pair[compare]
                    else:
                        player, worse, better = code_to_pair[abs(compare)]
                    if player not in self.payoffs:
                        self.payoffs[player] = {}
                    if worse not in self.payoffs[player]:
                        self.payoffs[player][worse] = []
                    if better not in self.payoffs[player]:
                        self.payoffs[player][better] = []
                    self.payoffs[player][better].append(worse)
        return self.payoffs
    
    def check_for_cycle_is_pre_worst(self):
        if 0 not in self.payoffs:
            return []
        else:
            for player, payofs in self.payoffs.items():
                if len(payofs.get(-1, [1,1,1,1])) > 1:
                    return []
            return [self]


# a = DiGraph({1:[2,4],2:[3,5],3:[],4:[5,6],5:[4,7],6:[],7:[]},{1:0,2:1,3:3,4:2,5:1,6:6,7:7})
#
# Game(a, 1)