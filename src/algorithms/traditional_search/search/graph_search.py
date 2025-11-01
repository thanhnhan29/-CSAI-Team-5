from collections import deque
import heapq
from .node import Node

NO_SOLUTION_MSG = "❌ No solution found"

class GraphSearch:
    """
    Base class for graph search algorithms (analogous to ACO/PSO classes)
    Implements BFS, DFS, and A* search
    """
    def __init__(self, initial_state, goal_test, get_successors, heuristic=None):
        """
        Initialize search algorithm
        
        Args:
            initial_state: Starting state
            goal_test: Function that returns True if state is goal
            get_successors: Function that returns list of (action, next_state, cost) tuples
            heuristic: Function that estimates cost to goal (for A*)
        """
        self.initial_state = initial_state
        self.goal_test = goal_test
        self.get_successors = get_successors
        self.heuristic = heuristic if heuristic else lambda state: 0
        
        # Statistics
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_frontier_size = 0
        
    def bfs(self):
        """
        Breadth-First Search
        Uses queue (FIFO) - explores level by level
        """
        print("Running BFS...")
        frontier = deque([Node(self.initial_state)])
        explored = set()
        self.nodes_generated = 1
        
        while frontier:
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            node = frontier.popleft()  # FIFO
            
            if self.goal_test(node.state):
                print(f"✅ Solution found at depth {node.depth}")
                print(f"📊 Nodes expanded: {self.nodes_expanded}, generated: {self.nodes_generated}")
                print(f"📊 Max frontier size: {self.max_frontier_size}")
                return node
            
            state_key = str(node.state)
            if state_key not in explored:
                explored.add(state_key)
                self.nodes_expanded += 1
                
                for action, next_state, cost in self.get_successors(node.state):
                    child = Node(
                        state=next_state,
                        parent=node,
                        action=action,
                        path_cost=node.path_cost + cost
                    )
                    frontier.append(child)
        print(NO_SOLUTION_MSG)
        return None
    
    def dfs(self, max_depth=100):
        """
        Depth-First Search
        Uses stack (LIFO) - explores deeply before backtracking
        """
        print("Running DFS...")
        frontier = [Node(self.initial_state)]  # Use list as stack
        explored = set()
        self.nodes_generated = 1
        
        while frontier:
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            node = frontier.pop()  # LIFO
            
            if self.goal_test(node.state):
                print(f"✅ Solution found at depth {node.depth}")
                print(f"📊 Nodes expanded: {self.nodes_expanded}, generated: {self.nodes_generated}")
                print(f"📊 Max frontier size: {self.max_frontier_size}")
                return node
            
            state_key = str(node.state)
            if state_key not in explored and node.depth < max_depth:
                explored.add(state_key)
                self.nodes_expanded += 1
                
                for action, next_state, cost in self.get_successors(node.state):
                    child = Node(
                        state=next_state,
                        parent=node,
                        action=action,
                        path_cost=node.path_cost + cost
                    )
                    frontier.append(child)
        print(NO_SOLUTION_MSG)
        return None
    
    def a_star(self):
        """
        A* Search
        Uses priority queue ordered by f(n) = g(n) + h(n)
        Optimal if heuristic is admissible
        """
        print("Running A*...")
        initial_node = Node(
            state=self.initial_state,
            heuristic_cost=self.heuristic(self.initial_state)
        )
        frontier = [initial_node]
        heapq.heapify(frontier)
        explored = set()
        self.nodes_generated = 1
        
        # Track best cost to reach each state
        best_cost = {str(self.initial_state): 0}
        
        while frontier:
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            node = heapq.heappop(frontier)
            
            if self.goal_test(node.state):
                print(f"✅ Solution found at depth {node.depth}")
                print(f"📊 Nodes expanded: {self.nodes_expanded}, generated: {self.nodes_generated}")
                print(f"📊 Max frontier size: {self.max_frontier_size}")
                return node
            
            state_key = str(node.state)
            if state_key not in explored:
                explored.add(state_key)
                self.nodes_expanded += 1
                
                for action, next_state, cost in self.get_successors(node.state):
                    new_cost = node.path_cost + cost
                    next_state_key = str(next_state)
                    
                    # Only add if we found a better path
                    if next_state_key not in best_cost or new_cost < best_cost[next_state_key]:
                        best_cost[next_state_key] = new_cost
                        child = Node(
                            state=next_state,
                            parent=node,
                            action=action,
                            path_cost=new_cost,
                            heuristic_cost=self.heuristic(next_state)
                        )
                        heapq.heappush(frontier, child)
                        self.nodes_generated += 1
        
        print("❌ No solution found")
        return None
