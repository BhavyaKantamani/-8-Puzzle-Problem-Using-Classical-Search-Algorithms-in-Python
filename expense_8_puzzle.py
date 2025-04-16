import sys
import heapq
from collections import deque
from datetime import datetime

class Node:
    def __init__(self, state, parent=None, action=None, cost=0, depth=0, f_value=0):
        self.state = state  
        self.parent = parent 
        self.action = action  
        self.cost = cost  
        self.depth = depth  
        self.f_value = f_value 

    def __lt__(self, other):
        return self.cost < other.cost  

    def __repr__(self):
        return f"< state = {self.state}, action = {{{self.action}}} g(n) = {self.cost}, d = {self.depth}, f(n) = {self.f_value}, Parent = Pointer to {self.parent} >"


def read_puzzle(filename):
    puzzle = []
    with open(filename, 'r') as file:
        for line in file:
            
            if "END" in line or not line.strip():
                continue
            puzzle.append(list(map(int, line.split())))
    return puzzle

def is_goal(state, goal):
    return state == goal


def dump_trace(fringe, closed_set, nodes_expanded, nodes_generated, start_file, goal_file, method, dump_flag):
    trace_file = f"trace-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"
    with open(trace_file, 'w') as file:
        command_line_args = [start_file, goal_file, method, str(dump_flag).lower()]
        file.write(f"Command-Line Arguments : {command_line_args}\n")
        file.write(f"Method Selected: {method}\n")
        file.write(f"Running {method}\n")

        for node in fringe:
            file.write(f"Generating successors to {node}:\n")
            successors = get_neighbors(node.state)
            file.write(f"\t{len(successors)} successors generated\n")

            file.write(f"\tClosed: {list(closed_set)}\n")
            file.write(f"\tFringe: [\n")
            for n in fringe:
                file.write(f"\t\t{n}\n")
            file.write("\t]\n")

        file.write(f"Nodes Popped: {nodes_expanded}\n")
        file.write(f"Nodes Expanded: {nodes_expanded}\n")
        file.write(f"Nodes Generated: {nodes_generated}\n")
        file.write(f"Max Fringe Size: {len(fringe)}\n")
    print(f"Trace dumped to {trace_file}")
def print_steps(path):
    print("Steps:")
    for step in path:
        print(f"\t{step}")

def get_direction(blank_pos, new_blank_pos):
    i, j = blank_pos
    ni, nj = new_blank_pos
    if ni == i and nj == j + 1:
        return "Left"
    elif ni == i and nj == j - 1:
        return "Right"
    elif ni == i + 1 and nj == j:
        return "Up"
    elif ni == i - 1 and nj == j:
        return "Down"
    return ""

def retrieve_path(node):
    path = []
    current = node
    while current.parent is not None:
        path.append(current.action)
        current = current.parent
    path.reverse()
    return path
def get_neighbors(state):
    neighbors = []
    blank_pos = [(i, row.index(0)) for i, row in enumerate(state) if 0 in row][0]
    i, j = blank_pos
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < 3 and 0 <= nj < 3:
            new_state = [row[:] for row in state]
            new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
            direction = get_direction((i, j), (ni, nj))
            neighbors.append((new_state, state[ni][nj], f"Move {state[ni][nj]} {direction}"))
    return neighbors

def manhattan_distance(state, goal):
    distance = 0
    for i in range(3):
        for j in range(3):
            value = state[i][j]
            if value != 0:  
                goal_i, goal_j = [(r, c) for r, row in enumerate(goal) for c, x in enumerate(row) if x == value][0]
                # Manhattan distance is the sum of absolute row and column differences
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance


def bfs(start, goal, dump_flag=False):
    start_node = Node(state=start, action="Start", cost=0, depth=0, f_value=0)
    fringe = deque([start_node])
    closed_set = set()
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe_size = 1

    while fringe:
        node = fringe.popleft()
        nodes_expanded += 1

        if is_goal(node.state, goal):
            print(f"Nodes Popped: {nodes_expanded}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Nodes Generated: {nodes_generated}")
            print(f"Max Fringe Size: {max_fringe_size}")
            print(f"Solution Found at depth {node.depth} with cost of {node.cost}.")
            print_steps(retrieve_path(node))
            if dump_flag:
                dump_trace(fringe, closed_set, nodes_expanded, nodes_generated, sys.argv[1], sys.argv[2], 'bfs', dump_flag)
            return

        closed_set.add(tuple(map(tuple, node.state)))

        for neighbor, move_cost, move in get_neighbors(node.state):
            if tuple(map(tuple, neighbor)) not in closed_set:
                new_node = Node(
                    state=neighbor,
                    parent=node,
                    action=move,
                    cost=node.cost + move_cost,
                    depth=node.depth + 1,
                    f_value= 0 
                )
                fringe.append(new_node)
                nodes_generated += 1

        max_fringe_size = max(max_fringe_size, len(fringe))

    if dump_flag:
        dump_trace(fringe, closed_set, nodes_expanded, nodes_generated, sys.argv[1], sys.argv[2], 'bfs', dump_flag)

    print("No solution found.")
    

def ucs(start, goal, dump_flag=False):
    start_node = Node(state=start, action="Start", cost=0, depth=0, f_value=0)
    fringe = [(start_node.cost, start_node)]  # (cost, node) tuples for priority queue
    closed_set = set()
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe_size = 1

    while fringe:
        _, node = heapq.heappop(fringe)
        nodes_expanded += 1

        if is_goal(node.state, goal):
            print(f"Nodes Popped: {nodes_expanded}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Nodes Generated: {nodes_generated}")
            print(f"Max Fringe Size: {max_fringe_size}")
            print(f"Solution Found at depth {node.depth} with cost of {node.cost}.")
            print_steps(retrieve_path(node))
            if dump_flag:
                dump_trace([n for _, n in fringe], closed_set, nodes_expanded, nodes_generated, sys.argv[1], sys.argv[2], 'ucs', dump_flag)
            return

        closed_set.add(tuple(map(tuple, node.state)))

        for neighbor, move_cost, move in get_neighbors(node.state):
            if tuple(map(tuple, neighbor)) not in closed_set:
                new_node = Node(
                    state=neighbor,
                    parent=node,
                    action=move,
                    cost=node.cost + move_cost, 
                    depth=node.depth + 1,
                    f_value=node.cost + move_cost 
                )
                heapq.heappush(fringe, (new_node.cost, new_node))  # Prioritize by cost
                nodes_generated += 1

        max_fringe_size = max(max_fringe_size, len(fringe))

    if dump_flag:
        dump_trace([n for _, n in fringe], closed_set, nodes_expanded, nodes_generated, sys.argv[1], sys.argv[2], 'ucs', dump_flag)

    print("No solution found.")
    
def dfs(start, goal, dump_flag=False):
    start_node = Node(state=start, action="Start", cost=0, depth=0, f_value=0)
    fringe = [start_node] 
    closed_set = set()
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe_size = 1

    while fringe:
        node = fringe.pop()  # LIFO order for DFS
        nodes_expanded += 1

        if is_goal(node.state, goal):
            print(f"Nodes Popped: {nodes_expanded}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Nodes Generated: {nodes_generated}")
            print(f"Max Fringe Size: {max_fringe_size}")
            print(f"Solution Found at depth {node.depth} with cost of {node.cost}.")
            print_steps(retrieve_path(node))
            if dump_flag:
                dump_trace(fringe, closed_set, nodes_expanded, nodes_generated, sys.argv[1], sys.argv[2], 'dfs', dump_flag)
            return

        closed_set.add(tuple(map(tuple, node.state)))

        for neighbor, move_cost, move in reversed(get_neighbors(node.state)):
            if tuple(map(tuple, neighbor)) not in closed_set:
                new_node = Node(
                    state=neighbor,
                    parent=node,
                    action=move,
                    cost=node.cost + move_cost,
                    depth=node.depth + 1,
                    f_value=node.cost + move_cost  
                )
                fringe.append(new_node)
                nodes_generated += 1

        max_fringe_size = max(max_fringe_size, len(fringe))

    if dump_flag:
        dump_trace(fringe, closed_set, nodes_expanded, nodes_generated, sys.argv[1], sys.argv[2], 'dfs', dump_flag)

    print("No solution found.")
    
def dls(start, goal, depth_limit, dump_flag=False):
    start_node = Node(state=start, action="Start", cost=0, depth=0, f_value=0)
    fringe = [start_node]  
    closed_set = set()
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe_size = 1

    while fringe:
        node = fringe.pop()  
        nodes_expanded += 1

        if is_goal(node.state, goal):
            print(f"Nodes Popped: {nodes_expanded}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Nodes Generated: {nodes_generated}")
            print(f"Max Fringe Size: {max_fringe_size}")
            print(f"Solution Found at depth {node.depth} with cost of {node.cost}.")
            print_steps(retrieve_path(node))
            if dump_flag:
                dump_trace(fringe, closed_set, nodes_expanded, nodes_generated, sys.argv[1], sys.argv[2], 'dls', dump_flag)
            return

        if node.depth < depth_limit:
            closed_set.add(tuple(map(tuple, node.state)))

            for neighbor, move_cost, move in reversed(get_neighbors(node.state)):
                if tuple(map(tuple, neighbor)) not in closed_set:
                    new_node = Node(
                        state=neighbor,
                        parent=node,
                        action=move,
                        cost=node.cost + move_cost,
                        depth=node.depth + 1,
                        f_value=node.cost + move_cost
                    )
                    fringe.append(new_node)
                    nodes_generated += 1

        max_fringe_size = max(max_fringe_size, len(fringe))

    if dump_flag:
        dump_trace(fringe, closed_set, nodes_expanded, nodes_generated, sys.argv[1], sys.argv[2], 'dls', dump_flag)

    print("No solution found.")
def ids(start, goal, dump_flag=False):
    depth_limit = 0
    nodes_expanded_total = 0
    nodes_generated_total = 0
    max_fringe_size_total = 0

    while True:
        print(f"Running DLS with depth limit {depth_limit}")
        result = dls_for_ids(start, goal, depth_limit, dump_flag)

      
        if result is not None:
            return result

      
        depth_limit += 1

def dls_for_ids(start, goal, depth_limit, dump_flag=False):
    start_node = Node(state=start, action="Start", cost=0, depth=0, f_value=0)
    fringe = [start_node]  
    closed_set = set()
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe_size = 1

    while fringe:
        node = fringe.pop()
        nodes_expanded += 1

        if is_goal(node.state, goal):
            print(f"Nodes Popped: {nodes_expanded}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Nodes Generated: {nodes_generated}")
            print(f"Max Fringe Size: {max_fringe_size}")
            print(f"Solution Found at depth {node.depth} with cost of {node.cost}.")
            print_steps(retrieve_path(node))
            if dump_flag:
                dump_trace(fringe, closed_set, nodes_expanded, nodes_generated, sys.argv[1], sys.argv[2], 'ids', dump_flag)
            return True  

        if node.depth < depth_limit:
            closed_set.add(tuple(map(tuple, node.state)))

            for neighbor, move_cost, move in reversed(get_neighbors(node.state)):
                if tuple(map(tuple, neighbor)) not in closed_set:
                    new_node = Node(
                        state=neighbor,
                        parent=node,
                        action=move,
                        cost=node.cost + move_cost,
                        depth=node.depth + 1,
                        f_value=node.cost + move_cost
                    )
                    fringe.append(new_node)
                    nodes_generated += 1

        max_fringe_size = max(max_fringe_size, len(fringe))

    return None  


def a_star(start, goal, dump_flag=False):
    start_node = Node(state=start, action="Start", cost=0, depth=0, f_value=manhattan_distance(start, goal))
    fringe = [(start_node.f_value, start_node)]  # Priority queue with f(n) = g(n) + h(n)
    closed_set = set()
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe_size = 1

    while fringe:
        _, node = heapq.heappop(fringe)  # Pop the node with the lowest f(n)
        nodes_expanded += 1

        if is_goal(node.state, goal):
            print(f"Nodes Popped: {nodes_expanded}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Nodes Generated: {nodes_generated}")
            print(f"Max Fringe Size: {max_fringe_size}")
            print(f"Solution Found at depth {node.depth} with cost of {node.cost}.")
            print_steps(retrieve_path(node))
            if dump_flag:
                dump_trace([n for _, n in fringe], closed_set, nodes_expanded, nodes_generated, sys.argv[1], sys.argv[2], 'a*', dump_flag)
            return

        closed_set.add(tuple(map(tuple, node.state)))

        neighbors = get_neighbors(node.state)
        for neighbor, move_cost, move in neighbors:
            if tuple(map(tuple, neighbor)) not in closed_set:
                new_node = Node(
                    state=neighbor,
                    parent=node,
                    action=move,
                    cost=node.cost + move_cost,  
                    depth=node.depth + 1,
                    f_value=node.cost + move_cost + manhattan_distance(neighbor, goal)  
                )
                heapq.heappush(fringe, (new_node.f_value, new_node)) 
                nodes_generated += 1

        max_fringe_size = max(max_fringe_size, len(fringe))  

    if dump_flag:
        dump_trace([n for _, n in fringe], closed_set, nodes_expanded, nodes_generated, sys.argv[1], sys.argv[2], 'a*', dump_flag)

    print("No solution found.")
    
def greedy_search(start, goal, dump_flag=False):
    start_node = Node(state=start, action="Start", cost=0, depth=0, f_value=manhattan_distance(start, goal))
    fringe = [(start_node.f_value, start_node)] 
    closed_set = set()
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe_size = 1

    while fringe:
        _, node = heapq.heappop(fringe) 
        nodes_expanded += 1

        if is_goal(node.state, goal):
            print(f"Nodes Popped: {nodes_expanded}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Nodes Generated: {nodes_generated}")
            print(f"Max Fringe Size: {max_fringe_size}")
            print(f"Solution Found at depth {node.depth} with cost of {node.cost}.")
            print_steps(retrieve_path(node))
            if dump_flag:
                dump_trace([n for _, n in fringe], closed_set, nodes_expanded, nodes_generated, sys.argv[1], sys.argv[2], 'greedy', dump_flag)
            return

        closed_set.add(tuple(map(tuple, node.state)))

        neighbors = get_neighbors(node.state)
        for neighbor, move_cost, move in neighbors:
            if tuple(map(tuple, neighbor)) not in closed_set:
                new_node = Node(
                    state=neighbor,
                    parent=node,
                    action=move,
                    cost=node.cost + move_cost, 
                    depth=node.depth + 1,
                    f_value=manhattan_distance(neighbor, goal) 
                )
                heapq.heappush(fringe, (new_node.f_value, new_node))  
                nodes_generated += 1

        max_fringe_size = max(max_fringe_size, len(fringe)) 

    if dump_flag:
        dump_trace([n for _, n in fringe], closed_set, nodes_expanded, nodes_generated, sys.argv[1], sys.argv[2], 'greedy', dump_flag)

    print("No solution found.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: expense_8_puzzle.py <start-file> <goal-file> <method> <dump-flag>")
        sys.exit(1)

    start_file = sys.argv[1]
    goal_file = sys.argv[2]
    method = sys.argv[3] if len(sys.argv) > 3 else 'a*'
    dump_flag = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else False

    start = read_puzzle(start_file)
    goal = read_puzzle(goal_file)

    if method == 'bfs':
        bfs(start, goal, dump_flag)
    elif method == 'ucs':
        ucs(start, goal, dump_flag)
    elif method == 'dfs':
        dfs(start, goal, dump_flag)
    elif method == 'dls':
        depth_limit = int(input("Enter Depth Limit: "))
        dls(start, goal, depth_limit, dump_flag)
    elif method == 'ids':
        ids(start, goal, dump_flag)
    elif method == 'greedy':
        greedy_search(start, goal, dump_flag)
    elif method == 'a*':
        a_star(start, goal, dump_flag)
    else:
        print(f"Method {method} not implemented yet.")