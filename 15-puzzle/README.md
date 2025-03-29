# 15 Puzzle Solver

## Overview
A 15 puzzle solver leveraging A* and ID A* algorithms, accompanied by an animation demonstration of the solving process. Authored by Yodeeshi from the School of Computer Science at Sun Yat - sen University.

## Features
- **Random Puzzle Generation**: Capable of producing solvable random 15 puzzles.
- **Algorithm - Based Solving**: Supports puzzle resolution using A* and ID A* algorithms.
- **Animation Playback**: Enables users to opt for an animated display of the puzzle - solving journey.

## Code Structure
| File | Function |
| --- | --- |
| `generator.py` | Generate puzzles, verify solvability, and handle puzzle - integer conversions. |
| `heuristics.py` | Compute various heuristic values for the puzzles. |
| `a_star.py` | Implement the A* algorithm for puzzle solving. |
| `id_a_star.py` | Implement the ID A* algorithm for puzzle solving. |
| `utils.py` | Retrieve the position of the blank tile and neighboring states. |
| `main.py` | Serve as the program's entry point, facilitating user interaction. |
| `visualizer.py` | Play the animation of the puzzle - solving process. |

## Running Steps
1. **Install Dependencies**: Install the `pygame` library using `pip install pygame`.
2. **Execute the Program**: Run the `main.py` file with `python main.py`.
3. **Make a Selection**: Input `1` to use the A* algorithm, `2` for the ID A* algorithm, or `exit` to terminate the program.
4. **Animate the Solution**: If a solution is found, enter `yes` to play the animation.

## Notes
- Depending on your operating system, additional libraries may be required to run the animation smoothly. Install `pywin32` on Windows, `pyobjc` on macOS, and `xdotool` on Linux if necessary.
- Solving complex puzzles can be time - consuming. Please exercise patience.

**Copyright**: This project is created by Yodeeshi from the School of Computer Science and Engineering, Sun Yat - sen University. All rights reserved for learning and research purposes. 
