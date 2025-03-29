from functools import lru_cache
from heuristics import mixed_heuristic
from utils import get_neighbors, goal_state_int
from generator import int_to_puzzle, puzzle_to_int

def ida_star(initial_state_int):
    @lru_cache(maxsize=None)
    def heuristic(state_int):
        return mixed_heuristic(int_to_puzzle(state_int))

    def search(path, g, bound, visited):
        node_int = path[-1]
        if node_int in visited:
            return float('inf')
        visited.add(node_int)

        f = g + heuristic(node_int)
        if f > bound:
            return f
        if node_int == goal_state_int:
            return 0

        min_bound = float('inf')
        for neighbor in get_neighbors(int_to_puzzle(node_int)):
            neighbor_int = puzzle_to_int(neighbor)
            if neighbor_int in path:
                continue
            path.append(neighbor_int)
            t = search(path, g + 1, bound, visited.copy())  # 传递副本避免共享状态
            if t == 0:
                return 0
            if t < min_bound:
                min_bound = t
            path.pop()

        return min_bound

    bound = heuristic(initial_state_int)
    path = [initial_state_int]
    visited = set()

    while True:
        visited.clear()
        t = search(path, 0, bound, visited)
        if t == 0:
            return [int_to_puzzle(state) for state in path]
        if t == float('inf'):
            return None
        # 调整边界值更新策略（避免过快增长）
        bound = max(1, t * 1.2)  # 适当增加步长