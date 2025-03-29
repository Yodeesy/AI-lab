import random

# 判断拼图是否可解
def is_solvable(puzzle):
    flat_puzzle = [num for row in puzzle for num in row]
    inversions = 0

    for i in range(len(flat_puzzle)):
        for j in range(i + 1, len(flat_puzzle)):
            if flat_puzzle[i] and flat_puzzle[j] and flat_puzzle[i] > flat_puzzle[j]:
                inversions += 1

    blank_row = next(i for i, row in enumerate(puzzle) if 0 in row)

    # 公式：如果空白格在倒数奇数行，逆序对数必须是偶数，否则必须是奇数
    return (inversions % 2 == 0) if (4 - blank_row) % 2 == 1 else (inversions % 2 == 1)

# 生成可解的随机拼图
def generate_puzzle():
    while True:
        puzzle = list(range(1, 16)) + [0]
        random.shuffle(puzzle)
        puzzle = [puzzle[i:i + 4] for i in range(0, 16, 4)]
        if is_solvable(puzzle):
            return puzzle

# 将 4x4 拼图转换为 64 位整数
def puzzle_to_int(puzzle):
    value = 0
    for row in puzzle:
        for num in row:
            value = (value << 4) | num  # 每个数字占 4 位
    return value

# 将 64 位整数转换回 4x4 拼图
def int_to_puzzle(value):
    puzzle = []
    for _ in range(4):
        row = []
        for _ in range(4):
            row.insert(0, value & 0xF)  # 取低 4 位
            value >>= 4
        puzzle.insert(0, row)  # 从底部插入，恢复顺序
    return puzzle
