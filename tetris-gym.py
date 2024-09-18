import cv2
import gymnasium as gym

#our input vector is [x change -4 to 5 (negatives move left, positive move right), rotate 0 to 3 (rotate CC) or negative (rotate clockwise)],  
vectors = [[4,0]]

from tetris_gymnasium.envs.tetris import Tetris

if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.reset(seed=42)
    i = 0
    terminated = False
    while not terminated:
        env.render()
        
        moveX, rotate = vectors[i % 1]
        if moveX < 0:
            for i in range(abs(moveX)):
                action = env.unwrapped.actions.move_left
                observation, reward, terminated, truncated, info = env.step(action)
        else:
            for i in range(moveX):
                action = env.unwrapped.actions.move_right
                observation, reward, terminated, truncated, info = env.step(action)
        i += 1

        key = cv2.waitKey(100) # timeout to see the movement
    print("Game Over!")