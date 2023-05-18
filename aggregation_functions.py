from itertools import product
import Game_class
from useful_funcs import powerset
import copy

def aggregation_func(curr_graph: dict, vertices,players):
    outpt = []
    for vertex, connected in curr_graph.items(): #####надо убрать эту вхаркоженую проверку на связность
        if len(connected) == 0:
            return outpt            
    for colors in product(list(range(players)), repeat = vertices):
        if len(set(colors)) < players:
            continue
        cls = dict(zip(list(range(vertices)), colors))
        a = Game_class.DiGraph(aj_dict=curr_graph,colors=cls)
        for v0 in range(vertices):
            g = Game_class.Game(a, v0)
            g.make_CNF()
            g.get_compares()
            outpt += g.check_for_cycle_is_pre_worst()
    return outpt

def correct_game_graph(curr_graph: dict, vertices,players):
    outpt = []
    for vertex, connected in curr_graph.items(): #####надо убрать эту вхаркоженую проверку на связность
        if len(connected) == 0:
            return outpt           
    for colors in product(list(range(players)), repeat = vertices):
        if len(set(colors)) < players:
            continue
        cls = dict(zip(list(range(vertices)), colors))
        a = Game_class.DiGraph(aj_dict=curr_graph,colors=cls)
        for v0 in range(vertices):
            g = Game_class.Game(a, v0)
            g.make_CNF()
            g.get_compares()
            outpt += [g] if g.payoffs else []
    return outpt

def correct_game_graph_with_outputs(curr_graph: dict, vertices,players):
    outpt = []
    for vertex, connected in curr_graph.items(): #####надо убрать эту вхаркоженую проверку на связность
        if len(connected) == 0:
            return outpt
    for vertex_set in powerset(curr_graph.keys()):
        updated_graph = copy.deepcopy(curr_graph)
        last_vertex_num = vertices
        if len(vertex_set) == 0:
            continue
        for vertex in vertex_set:
            updated_graph[vertex].append(last_vertex_num)
            updated_graph[last_vertex_num] = []
            last_vertex_num += 1
        for colors in product(list(range(players)), repeat = vertices):
            if len(set(colors)) < players:
                continue
            # print(colors)
            cls = dict(zip(list(range(last_vertex_num)), list(colors)+([players+1]*(last_vertex_num  - len(colors)))))
            # print(cls, updated_graph)
            a = Game_class.DiGraph(aj_dict=updated_graph,colors=cls)
            for v0 in range(vertices):
                g = Game_class.Game(a, v0)
                g.make_CNF()
                g.get_compares()
                outpt += [g] if g.payoffs else []
    return outpt


def final_mega_check_with_outputs(curr_graph: dict, vertices,players):
    outpt = []
    for vertex, connected in curr_graph.items(): #####надо убрать эту вхаркоженую проверку на связность
        if len(connected) == 0:
            return outpt
    updated_graph = curr_graph.copy()
    last_vertex_num = vertices
    for vertex, connected in curr_graph.items(): #####надо убрать эту вхаркоженую проверку на связность
        updated_graph[vertex].append(last_vertex_num)
        updated_graph[last_vertex_num] = []
        last_vertex_num += 1
    for colors in product(list(range(players)), repeat = vertices):
        if len(set(colors)) < players:
            continue
        cls = dict(zip(list(range(last_vertex_num)), colors + colors))
        # print(cls, updated_graph)
        a = Game_class.DiGraph(aj_dict=updated_graph,colors=cls)
        for v0 in range(vertices):
            g = Game_class.Game(a, v0)
            g.make_CNF()
            g.get_compares()
            outpt += g.check_for_cycle_is_pre_worst()
    return outpt
