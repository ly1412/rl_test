from envs.maze_env import Maze


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

    print('end of game')
    env.destory()


if __name__ == '__main__':
    env = Maze()
    RL = QLearningTale(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()