from agent import Agent

'''
Mode 1: Tetris gymnasium website had reward mappings as such (Bumpiness on)
    - siterewardtrain.tmp
            alife: 0.001
            clear_line = 1
            game_over = -2
            // just keep invalid action the same
Mode 2: After installing tetris gymnasium and looking at its reward mappings, I found (Bumpiness on)
    - downloadrewardtrain.tmp
            alife: 1
            clear_line = 1
            game_over = 0
            invalid_action = -0.1
Mode 3: Mode 1 but with no bumpiness added to the reward system
    - nobumpytrain.tmp

To change the modes, first go into the tetris code and change the rewards to the corresponding
mode that you want
'''
agent = Agent(mode=3, starting_epsilon=0.75, num_episodes=100)
agent.run(is_training=True, render=True)
