import cv2
import gymnasium as gym

#our input vector is [x change -4 to 5 (negatives move left, positive move right), rotate 0 to 3 (rotate CC) or negative (rotate clockwise)],  
vectors = [[2,0],[-4,0]]

from tetris_gymnasium.envs.tetris import Tetris

if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.reset(seed=42)
    i = 0
    terminated = False
    action = env.unwrapped.actions.move_left
    env.render()
    observation, reward, terminated, truncated, info = env.step(action)
    
    
    print(observation)

    key = cv2.waitKey(0) # timeout to see the movement
    print("Game Over!")