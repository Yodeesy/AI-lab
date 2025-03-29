def manhattan_distance(puzzle):
    distance = 0
    for i in range(4):
        for j in range(4):
            value = puzzle[i][j]
            if value != 0:
                target_x, target_y = (value - 1) // 4, (value - 1) % 4
                distance += abs(target_x - i) + abs(target_y - j)
    return distance

def count_inversions(puzzle):
    flat_puzzle = [num for row in puzzle for num in row if num != 0]
    inversions = sum(1 for i in range(len(flat_puzzle)) for j in range(i + 1, len(flat_puzzle)) if flat_puzzle[i] > flat_puzzle[j])
    return inversions

def linear_conflict(puzzle):
    conflicts = 0
    for i in range(4):
        row_values = [(puzzle[i][j], j) for j in range(4) if puzzle[i][j] != 0]
        sorted_values = sorted(row_values, key=lambda x: x[0])
        for j in range(len(row_values)):
            if row_values[j][1] > sorted_values[j][1]:
                conflicts += 1
    for j in range(4):
        col_values = [(puzzle[i][j], i) for i in range(4) if puzzle[i][j] != 0]
        sorted_values = sorted(col_values, key=lambda x: x[0])
        for i in range(len(col_values)):
            if col_values[i][1] > sorted_values[i][1]:
                conflicts += 1
    return 2 * conflicts

def mixed_heuristic(puzzle, alpha=1.0, beta=0.4, gamma=0.8):
    return alpha * manhattan_distance(puzzle) + beta * count_inversions(puzzle) + gamma * linear_conflict(puzzle)