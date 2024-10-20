import gymnasium as gym

from deterministic_mcts import DeterministicMCTS

'''
Map:

SFFF
FHFH
FFFH
HFFG
'''
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
(init_state, _) = env.reset()

# Curr problem = difficult to simulate using gymnasium because you cannot start from the middle
# Convert this to Monte Carlo Learning instead
tree = DeterministicMCTS(init_state)
curr_state = init_state
end = False
while not end:
    tree.search(env)
    tree.print()
    next_action = tree.choose()
    (_, _, terminaed, truncated, _) = env.step(next_action)
    end = terminaed or truncated
env.close()
