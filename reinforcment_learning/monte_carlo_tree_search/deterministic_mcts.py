import math

from gymnasium import Env

class Node():
    def __init__(self, state):
        self.state = state
        self.wins = 0
        self.num_visits = 0
        self.parent = None
        self.children: dict = {}
    
    def estimated_utility(self) -> int:
        if self.num_visits == 0:
            return -math.inf
        return self.wins / self.num_visits
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0

class DeterministicMCTS():
    def __init__(self, init_state):
        self.root_node = Node(init_state)
        self.curr_node = self.root_node
        self.actions_taken = []

    def choose(self):
        best_action = self._get_max_utility_action(self.curr_node)
        self.curr_node = self.curr_node.children[best_action]
        self.actions_taken = []
        return best_action
    
    def print(self):
        self._print_node(self.root_node)
    
    def _print_node(self, node: Node):
        print(f'{node.state}: {node.wins}/{node.num_visits}')

        for child in node.children:
            self._print_node(child)
    
    def search(self, env: Env, num_iter = 50):
        for _ in range(num_iter):
            node = self._selection(self.curr_node)
            node = self._expansion(env, node)
            reward = self._simulation(env, node)
            self._backup(node, reward)

    def _selection(self, node: Node) -> Node:
        curr_node = node
        while not curr_node.is_leaf():
            max_utility = -math.inf
            argmax_utility = 0
            for action, child in curr_node.children:
                # If child not visited, select child
                if child.num_visits == 0:
                    return child

                # If all children visited, select using UCT
                est_utility = self._uct(curr_node, child)
                if est_utility > max_utility:
                    max_utility = est_utility
                    argmax_utility = action

            curr_node = curr_node.children[argmax_utility]
        return curr_node

    def _expansion(self, env: Env, node: Node) -> Node:
        # If node has not been visited yet, no need to expand
        if node.num_visits == 0:
            return node

        # If node has been visited, instantiate child nodes and return first
        for action in range(env.action_space.n):
            node.children[action] = Node()
            node.children[action].parent = node
        return node.children[0]
    
    def _simulation(self, env: Env, node: Node) -> int:
        # Simulate starting from current node to end

        # From root node to current node
        next_state = 0
        curr_node = self.root_node
        while next_state != node.state:
            best_action = self._get_max_utility_action(curr_node)
            (next_state, _, _, _, _) = env.step(best_action)
            curr_node = curr_node.children[best_action]

        # From current node to end
        end = False
        while not end:
            next_action = env.action_space.sample()
            (_, reward, terminated, truncated, _) = env.step(next_action)
            end = terminated or truncated
        return reward
    
    def _get_max_utility_action(self, node: Node) -> int:
        max_utility = -math.inf
        argmax_utility = 0
        for action, child in node.children:
            est_utility = child.estimated_utility()
            if est_utility > max_utility:
                max_utility = est_utility
                argmax_utility = action
        return argmax_utility

    def _backup(self, node: Node, reward: int):
        curr_node = node
        while curr_node is not None:
            node.num_visits += 1
            node.wins += reward
            curr_node = curr_node.parent

    def _uct(self, node: Node, child: Node):
        # Constant balancing exploitation and exploration
        C = 1
        return child.estimated_utility() + C * math.sqrt(math.log(node.num_visits) / child.num_visits)
