from env import World
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    question = ['value_iteration', 'MC', 'sarsa', 'q_learning']  # 'value_iteration', 'MC', 'sarsa', 'q_learning', 'default'
    env = World()
    pis = {}
    qs = {}
    vs = {}
    if 'value_iteration' in question:
        all_values, all_actions, action_values = env.value_iteration()
        pi = np.eye(env.nActions)[all_actions[-1].astype(np.int8)]
        env.plot_policy(pi)
        env.plot_qvalue(action_values)
        env.plot_value(all_values[-1])
        vs['value_iteration_True'] = all_values[-1]
        qs['value_iteration_True'] = action_values
        pis['value_iteration_True'] = pi

        all_values, all_actions, action_values = env.value_iteration(synchronous=False)
        pi = np.eye(env.nActions)[all_actions[-1].astype(np.int8)]

        env.plot_policy(pi)
        env.plot_qvalue(action_values)
        env.plot_value(all_values[-1])
        vs['value_iteration_False'] = all_values[-1]
        qs['value_iteration_False'] = action_values
        pis['value_iteration_False'] = pi

    if 'MC' in question:
        for explore_strats, n_episodes, eps_D in [(True, 100 * 10 ** 3, 100), (False, 100 * 10 ** 3, 1000)]:
            pi, q = env.mc_control(n_episodes, lr=1.5e-3,
                                   glie=lambda x: 1 / (x / eps_D + 1),
                                   explore_starts=explore_strats)
            pi_opt = np.eye(env.nActions)[np.argmax(pi, axis=-1)]
            env.plot_policy(pi_opt)
            env.plot_qvalue(q)
            env.plot_value(q.max(axis=-1))
            pis[f'MC_{explore_strats}'] = pi_opt
            qs[f'MC_{explore_strats}'] = q
            vs[f'MC_{explore_strats}'] = q.max(axis=-1)
    if 'sarsa' in question:
        q, pi = env.sarsa(50 * 10 ** 3, lr=2.5e-3, glie=lambda x: 1 / (x / 1000 + 1))
        env.plot_policy(pi)
        env.plot_qvalue(q)
        env.plot_value(q.max(axis=-1))
        pis['sarsa'] = pi
        qs['sarsa'] = q
        vs['sarsa'] = q.max(axis=-1)

    if 'q_learning' in question:

        q, pi = env.q_learning(50 * 10 ** 3, lr=1.5e-3, glie=lambda x: 1 / (x / 100 + 1))

        env.plot_policy(pi)
        env.plot_qvalue(q)
        env.plot_value(q.max(axis=-1))
        pis['q_learning'] = pi
        qs['q_learning'] = q
        vs['q_learning'] = q.max(axis=-1)

    if 'default' in question:
        num_episodes = 10
        for n_episode in np.arange(0, num_episodes):

            env.reset()
            done = False
            t = 0
            env.show()
            while not done:
                env.render()
                action = np.random.randint(1, env.nActions)  # take a random action
                next_state, reward, done = env.step(action)  # observe next_state and reward

                env.close()
                t += 1
                if done:
                    print("Episode", n_episode + 1, "finished after {} timesteps".format(t + 1))
                    break
