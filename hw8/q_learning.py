import sys
from environment import MountainCar
import random
from typing import Dict, List
import numpy as np


class LinearModel:
    
    def __init__(self, state_size: int, action_size: int,
    lr: float, indices: bool):
        self.w = np.zeros((action_size,state_size ))
        self.b = 0.0
        self.lr = lr
        self.indices = indices

    def predict(self, state: Dict[int, int]) -> List[float]:
        q = np.sum(state * self.w, axis=1) + self.b
        return q

    def update(self, state: Dict[int, int], action: int, target: int):
        
        self.w[action] -= self.lr * (target)*state
        self.b -=  self.lr * (target)


class QLearningAgent:
    
    def __init__(self, env: MountainCar, mode: str = None, gamma: float = 0.9,
    lr: float = 0.01, epsilon:float = 0.05):
        self.env = env
        self.state_size = env.state_space
        self.action_size = env.action_space
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.model = LinearModel(self.state_size, self.action_size, self.lr,indices=True if mode == "tile" else False)

    def get_action(self, state: Dict[int, int]) -> int:

        num = random.random()
        if num >=(1-epsilon):
            action = random.randint(0,2)
        else: 
            action = np.argmax(self.model.predict(state))
        return action

    # take greedy or xx action
    
    def matrix_generation(self,dict: Dict[int, int], size: int, indices: bool):
            array = np.zeros(size)
            if indices:
                for k in dict.keys():
                    array[k] = 1
            else:
                array = np.array(list(dict.values()))
            return array
    
    def train(self, episodes: int, max_iterations: int) -> List[float]:
        rewards =[]
        for i in range(episodes):
            state = self.env.reset()
            state = self.matrix_generation(state, self.state_size, indices=mode == "tile")
            current_reward = 0 
            for j in range(max_iterations):
                                 
                action = self.get_action(state)
                new_state, reward, done =  self.env.step(action)
                new_state = self.matrix_generation(new_state, self.state_size, indices=mode == "tile")
                q_next = self.model.predict(new_state)
                q_o = self.model.predict(state)[action]
                target = q_o - reward - self.gamma*(np.max(q_next))
                self.model.update(state, action, target)
                current_reward += reward
                state = new_state

                if done:
                	break

            rewards.append(current_reward)
                
        return rewards   


   
if __name__ == '__main__':

    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    lr = float(sys.argv[8])
    env = MountainCar(mode=mode)
    agent = QLearningAgent(env, mode=mode, gamma=gamma, epsilon=epsilon, lr=lr)
    returns = agent.train(episodes, max_iterations)
    np.savetxt(returns_out,returns)
    np.savetxt(weight_out, np.hstack([agent.model.b, agent.model.w.flatten(order='F')]))