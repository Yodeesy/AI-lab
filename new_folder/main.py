from generator import generate_puzzle, puzzle_to_int, int_to_puzzle
from a_star import a_star
from id_a_star import ida_star
import time
from visualizer import play_animation, quit_pygame

def main():
    puzzle = generate_puzzle()
    puzzle_int = puzzle_to_int(puzzle)
    print("Initial Puzzle:")
    for row in puzzle:
        print(row)

    try:
        while True:
            choice = input("Choice: 1 for A*, 2 for ID A*. Type 'exit' to quit. \n")

            if choice == "1":
                print("\n Solving with A* algorithm... (❁´◡`❁) \n")
                start_time = time.time()
                a_star_solution = a_star(puzzle_int)
                end_time = time.time()
                elapsed_time = end_time - start_time

                if a_star_solution:
                    print("Solution steps:")
                    for step, state in enumerate(a_star_solution, 1):  # 步骤从1开始计数
                        print(f"Step {step}:")
                        for row in state:
                            print(row)
                        print()
                    steps = len(a_star_solution) - 1  # 初始状态不算步数
                    print(f"Solution found in {elapsed_time:.2f} seconds ({steps} moves)")

                    play_choice = input("Do you want to play the animation? (yes/no) ")
                    if play_choice.lower() == "yes":
                        play_animation(a_star_solution)
                else:
                    print("(⊙_⊙;) No solution found.")

            elif choice == "2":
                print("\n Solving with ID A* algorithm... (❁´◡`❁) \n")
                start_time = time.time()
                id_a_star_solution = ida_star(puzzle_int)
                end_time = time.time()
                elapsed_time = end_time - start_time

                if id_a_star_solution:
                    print("Solution steps:")
                    for step, state in enumerate(id_a_star_solution, 1):
                        print(f"Step {step}:")
                        for row in state:
                            print(row)
                        print()
                    steps = len(id_a_star_solution) - 1
                    print(f"Solution found in {elapsed_time:.2f} seconds ({steps} moves)")

                    play_choice = input("Do you want to play the animation? (yes/no) ")
                    if play_choice.lower() == "yes":
                        play_animation(id_a_star_solution)
                else:
                    print("(⊙_⊙;) No solution found.")

            elif choice == "exit":
                print("\n(╥_╥) Goodbye! \n")
                break

            else:
                print("￣へ￣ Invalid choice.")
    finally:
        quit_pygame()

if __name__ == "__main__":
    main()