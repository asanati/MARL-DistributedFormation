from simulate import FormationSimulator
from pynput import keyboard
import matplotlib.pyplot as plt
import torch

num_agents = 3

agent_to_move = 0
def key_cb(key, env):
    if key == keyboard.Key.esc:
        plt.close('all')
        return
    action = torch.zeros(num_agents, 2)
    # check if key is a number between 0-4
    try:
        key = int(key.char)
        if 0 <= key < num_agents:
            global agent_to_move
            agent_to_move = key
            print(f"Moving agent {agent_to_move} from next move...")
    except (AttributeError, ValueError):
        pass

    speed = 10
    if key == keyboard.Key.up:
        action[agent_to_move, 1] = speed
    elif key == keyboard.Key.down:
        action[agent_to_move, 1] = -speed
    elif key == keyboard.Key.left:
        action[agent_to_move, 0] = -speed
    elif key == keyboard.Key.right:
        action[agent_to_move, 0] = speed
    else:
        return True
    
    obs, rewards, done, info = env.step(action)
    env.fig.canvas.draw_idle()
    print("-"*10)
    print(f"{action=}\n{obs=}\n{rewards=}\n{done=}\n{info=}")
    
    return True

if __name__ == "__main__":
    env = FormationSimulator(visualize=True, num_agents=num_agents, log=False)
    print("Press 0-4 to move the agent. Press ESC to exit.")
    print(f"press number between 0-{num_agents-1} to choose which agent to move")
    listener = keyboard.Listener(on_press=lambda key: key_cb(key, env))
    listener.start()
    plt.show()