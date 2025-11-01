class Node:
    """
    Represents a state in the search tree
    Each node represents a partial or complete solution
    """
    def __init__(self, state, parent=None, action=None, path_cost=0, heuristic_cost=0):
        """
        Initialize a search node
        
        Args:
            state: The state represented by this node (e.g., list of selected items)
            parent: Parent node in the search tree
            action: Action taken to reach this node from parent
            path_cost: Cost from root to this node (g(n))
            heuristic_cost: Estimated cost to goal (h(n))
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.heuristic_cost = heuristic_cost
        self.depth = 0 if parent is None else parent.depth + 1
        
    def get_f_cost(self):
        """Total cost for A*: f(n) = g(n) + h(n)"""
        return self.path_cost + self.heuristic_cost
    
    def get_path(self):
        """Return the sequence of states from root to this node"""
        path = []
        node = self
        while node:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))
    
    def get_actions(self):
        """Return the sequence of actions from root to this node"""
        actions = []
        node = self
        while node.parent:
            actions.append(node.action)
            node = node.parent
        return list(reversed(actions))
    
    def __lt__(self, other):
        """For priority queue comparison in A*"""
        return self.get_f_cost() < other.get_f_cost()
    
    def __eq__(self, other):
        """Check if two nodes have the same state"""
        if not isinstance(other, Node):
            return False
        return self.state == other.state
    
    def __hash__(self):
        """Allow nodes to be used in sets/dicts"""
        return hash(str(self.state))
    
    def __repr__(self):
        return f"Node(state={self.state}, cost={self.path_cost}, h={self.heuristic_cost})"
