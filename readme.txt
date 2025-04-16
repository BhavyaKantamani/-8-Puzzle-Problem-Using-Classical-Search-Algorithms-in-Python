AI-Assignment-1

Name: Bhavya Sri Kantamani
UTA ID: 1002118109

Programming Language:  
 Python 3.10 (tested on Python 3.10.6)

Code Structure:

1. expense_8_puzzle.py:  
   This file contains the implementation of various search algorithms to solve the modified 8-puzzle problem, also known as the Expense 8-puzzle problem.

   The following search algorithms are implemented:
   ◦ BFS (Breadth-First Search)
   ◦ UCS (Uniform Cost Search)
   ◦ DFS (Depth-First Search) 
   ◦ DLS (Depth-Limited Search) 
   ◦ IDS (Iterative Deepening Search)
   ◦ Greedy Search
   ◦ A* Search (This is the default method if no search method is specified)

2. Supporting Functions:
   ◦ read_puzzle(): Reads the puzzle input from a text file.
   ◦ manhattan_distance(): Calculates the Manhattan distance heuristic for A* and Greedy search.
   ◦ dump_trace(): Generates trace logs when the dump flag is set to true.
   ◦ get_neighbors(): Generates successors for a given puzzle state.
   ◦ retrieve_path(): Traces the steps to the solution.


How to Run the Code:

1. Python Setup:
   ◦Ensure Python 3.10 or higher is installed.
   ◦Ensure that the start and goal states are in the same directory.
   ◦Install any dependencies if required using pip. (No external libraries required for this code except Python's standard library)

2. Running the Program:
   ◦ The program should be run from the command line.
   ◦ Command Syntax:
      	python3 expense_8_puzzle.py <start-file> <goal-file> <method> <dump-flag>

   - Parameters:
     ◦ <start-file>: Text file containing the starting configuration of the puzzle.
     ◦ <goal-file>: Text file containing the goal configuration of the puzzle.
     ◦ <method>: The search algorithm to use. Available options are:
       ◦ bfs: Breadth-First Search
       ◦ ucs: Uniform Cost Search
       ◦ dfs: Depth-First Search 
       ◦ dls: Depth-Limited Search 
       ◦ ids: Iterative Deepening Search 
       ◦ greedy: Greedy Search
       ◦ a*: A* Search (default if no method is specified)
     ◦ <dump-flag>: true if you want to generate a trace file; otherwise, false.

3. Example Usage:
   	python3 expense_8_puzzle.py start.txt goal.txt bfs true

4. Trace File:
   ◦ If <dump-flag> is set to true, a trace file will be generated with the name format: trace-<date>-<time>.txt.
   ◦ The trace file contains detailed information about the search process, including the contents of the fringe and closed sets at each step.

5. Puzzle File Format:
   ◦ The start and goal files should contain a 3x3 grid of integers, with 0 representing the blank tile.
   Example:
     1 2 3
     4 0 5
     6 7 8
     END OF FILE

6. Compilation:
   ◦ No compilation is required since this is a Python implementation.


Instructions for Running on ACS Omega:

  ◦ The code is designed to run in a local Python 3.10 environment.
  ◦ If you wish to run the code on ACS Omega, ensure the correct Python version is available. You can run the script using the following command:
  	python3 expense_8_puzzle.py <start-file> <goal-file> <method> <dump-flag>



Note: 
1. Clear and specific instructions have been provided for running the code, and the program has been tested on Python 3.10. In case of any issues, please ensure the correct
   Python version is installed.

2. Generating the dump file may take longer than expected. The dump file size can reach more than 10 GB for some searches.



