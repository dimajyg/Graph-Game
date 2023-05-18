import Parallelisation
from itertools import combinations_with_replacement
import pickle

from aggregation_functions import aggregation_func
from aggregation_functions import correct_game_graph
from aggregation_functions import correct_game_graph_with_outputs
from aggregation_functions import final_mega_check_with_outputs

if __name__ == "__main__":
    result = Parallelisation.parallel_procces_graphs('/home/dtikhanovskii/Graph-Game/Graph-Game/Graph-Game/digraph_5.txt',5,4,correct_game_graph_with_outputs)
    with open("all_games_without_nash_eq_5_4players.pkl", "wb") as output:
        pickle.dump(result,output,pickle.HIGHEST_PROTOCOL)
        # output.write(str(list(map(lambda x: x.graph.aj_dict,result))))