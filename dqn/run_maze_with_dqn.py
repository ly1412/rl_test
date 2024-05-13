from envs.maze_env import Maze
from deep_q_work import DeepQNetwork


def update():
    step = 0
    for episode in range(300):
        observation = env.reset()
        print(episode)
        while True:
            env.render()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            observation = observation_
            if done:
                break

            step+=1

    print('end of game')
    # env.destory()


if __name__ == '__main__':
    env = Maze()
    # RL = SarsaTable(actions=list(range(env.n_actions)))
    RL = DeepQNetwork(actions=list(range(env.n_actions)),
                      n_features=env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000
                      )

    env.after(100, update)
    env.mainloop()
    RL.plot_cost()