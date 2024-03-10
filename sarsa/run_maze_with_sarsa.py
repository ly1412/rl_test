from envs.maze_env import Maze
from sarsa import SarsaTable


def update():
    for episode in range(100):
        observation = env.reset()
        print(observation)
        action = RL.choose_action(str(observation))
        while True:
            env.render()
            observation_, reward, done = env.step(action)
            action_ = RL.choose_action(str(observation_))
            RL.learn(str(observation), action, reward, str(observation_), action_)
            observation = observation_
            action = action_

            if done:
                break;
        if episode % 10 == 0:
            print(episode)

    print('end of game')
    env.destory()


if __name__ == '__main__':
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()