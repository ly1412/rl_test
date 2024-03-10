from envs.maze_env import Maze
from q_learning import QLearningTable


def update():
    for episode in range(100):
        observation = env.reset()
        print(observation)
        while True:
            env.render()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_

            if done:
                break;
        if episode % 10 == 0:
            print(RL.q_table)

    print('end of game')
    env.destory()


if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()