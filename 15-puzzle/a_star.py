import heapq
from functools import lru_cache
from heuristics import mixed_heuristic
from utils import get_neighbors, goal_state_int
from generator import int_to_puzzle, puzzle_to_int

@lru_cache(maxsize=None)
def heuristic(state_int):
    return mixed_heuristic(int_to_puzzle(state_int))

def a_star(initial_state_int):
    open_list = []
    closed_list = set()
    best_cost = float('inf')
    best_path = None

    heapq.heappush(open_list, (heuristic(initial_state_int), 0, initial_state_int, [initial_state_int]))

    while open_list:
        _, g, current_state_int, path = heapq.heappop(open_list)

        if current_state_int in closed_list:
            continue
        closed_list.add(current_state_int)

        if current_state_int == goal_state_int:
            if g < best_cost:
                best_cost = g
                best_path = [int_to_puzzle(state) for state in path]
            continue

        current_state = int_to_puzzle(current_state_int)
        for neighbor in get_neighbors(current_state):
            neighbor_int = puzzle_to_int(neighbor)  # 转换为整数
            if neighbor_int in closed_list:
                continue

            new_g = g + 1
            new_path = path + [neighbor_int]
            f = new_g + heuristic(neighbor_int)

            if f < best_cost:
                heapq.heappush(open_list, (f, new_g, neighbor_int, new_path))

    return best_path
