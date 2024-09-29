import gymnasium as gym

'''
Map:

SFFF
FHFH
FFFH
HFFG
'''
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array")
env.reset()

end = False
while not end:
    next_action = env.action_space.sample()
    (next_state, reward, terminaed, truncated, _) = env.step(next_action)
    print(f'{next_action} -> {next_state}, {reward}')
    end = terminaed or truncated
