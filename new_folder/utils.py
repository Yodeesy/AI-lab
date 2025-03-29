from generator import puzzle_to_int

goal_state = [[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 0]]
goal_state_int = puzzle_to_int(goal_state)

def get_blank_position(state):
    for i in range(4):
        for j in range(4):
            if state[i][j] == 0:
                return i, j

def get_neighbors(state):
    blank_row, blank_col = get_blank_position(state)
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dr, dc in moves:
        new_row, new_col = blank_row + dr, blank_col + dc
        if 0 <= new_row < 4 and 0 <= new_col < 4:
            new_state = [row[:] for row in state]
            new_state[blank_row][blank_col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[blank_row][blank_col]
            yield new_state