import numpy as np
import matplotlib.pyplot as plt
import _env
import progressbar


class World(_env.Hidden):

    def __init__(self):
        self.nRows = 4
        self.nCols = 5
        self.stateInitial = [4]
        self.stateTerminals = [1, 2, 10, 12, 17, 20]
        self.stateObstacles = []
        self.stateHoles = [1, 2, 10, 12, 20]
        self.stateGoal = [17]
        self.nStates = 20
        self.nActions = 4

        self.observation = [4]  # initial state

    @staticmethod
    def _truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def _plot_world(self):

        nRows = self.nRows
        nCols = self.nCols
        stateObstacles = self.stateObstacles
        stateTerminals = self.stateTerminals
        stateGoal = self.stateGoal
        coord = [[0, 0], [nCols, 0], [nCols, nRows], [0, nRows], [0, 0]]
        xs, ys = zip(*coord)
        plt.plot(xs, ys, "black")
        for i in stateObstacles:
            (I, J) = np.unravel_index(i - 1, shape=(nRows, nCols), order='F')
            I = I + 1
            J = J
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            plt.fill(xs, ys, "0.3")
            plt.plot(xs, ys, "black")
        for i in stateTerminals:
            # print("stateTerminal", i)
            (I, J) = np.unravel_index(i - 1, shape=(nRows, nCols), order='F')
            I = I + 1
            J = J
            # print("I,J = ", I,J)
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            # print("coord", xs,ys)
            plt.fill(xs, ys, "0.6")
            plt.plot(xs, ys, "black")
        for i in stateGoal:
            (I, J) = np.unravel_index(i - 1, shape=(nRows, nCols), order='F')
            I = I + 1
            J = J
            # print("I,J = ", I,J)
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            # print("coord", xs,ys)
            plt.fill(xs, ys, "0.9")
            plt.plot(xs, ys, "black")
        plt.plot(xs, ys, "black")
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        plt.plot(X, Y, 'k-')
        plt.plot(X.transpose(), Y.transpose(), 'k-')

    @staticmethod
    def _truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def plot(self):
        """
        plot function
        :return: None
        """
        nStates = self.nStates
        nRows = self.nRows
        nCols = self.nCols
        self._plot_world()
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.5, j - 0.5, str(states[k]), fontsize=26, horizontalalignment='center',
                         verticalalignment='center')
                k += 1
        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_value(self, valueFunction):

        """
        plot state value function V

        :param policy: vector of values of size nStates x 1
        :return: None
        """

        nStates = self.nStates
        nRows = self.nRows
        nCols = self.nCols
        stateObstacles = self.stateObstacles
        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateObstacles:
                    plt.text(i + 0.5, j - 0.5, str(self._truncate(np.round(valueFunction[k], 4), 3)), fontsize=16,
                             horizontalalignment='center', verticalalignment='center')
                k += 1
        # label states by numbers
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.92, j - 0.92, str(states[k]), fontsize=11, horizontalalignment='right',
                         verticalalignment='bottom')
                k += 1

        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_policy(self, policy):

        """
        plot (stochastic) policy

        :param policy: matrix of policy of size nStates x nActions
        :return: None
        """
        # remove values below 1e-6
        policy = policy * (np.abs(policy) > 1e-6)

        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateObstacles = self.stateObstacles
        stateTerminals = self.stateTerminals
        # policy = policy.reshape(nRows, nCols, order="F").reshape(-1, 1)
        # generate mesh for grid world
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        # generate locations for policy vectors
        # print("X = ", X)
        X1 = X.transpose()
        X1 = X1[:-1, :-1]
        # print("X1 = ", X1)
        Y1 = Y.transpose()
        Y1 = Y1[:-1, :-1]
        # print("Y1 =", Y1)
        X2 = X1.reshape(-1, 1) + 0.5
        # print("X2 = ", X2)
        Y2 = np.flip(Y1.reshape(-1, 1)) + 0.5
        # print("Y2 = ", Y2)
        # reshape to matrix
        X2 = np.kron(np.ones((1, nActions)), X2)
        # print("X2 after kron = ", X2)
        Y2 = np.kron(np.ones((1, nActions)), Y2)
        # print("X2 = ",X2)
        # print("Y2 = ",Y2)
        # define an auxiliary matrix out of [1,2,3,4]
        mat = np.cumsum(np.ones((nStates, nActions)), axis=1).astype("int64")
        # print("mat = ", mat)
        # if policy vector (policy deterministic) turn it into a matrix (stochastic policy)
        # print("policy.shape[1] =", policy.shape[1])
        if policy.shape[1] == 1:
            policy = (np.kron(np.ones((1, nActions)), policy) == mat)
            policy = policy.astype("int64")
            print("policy inside", policy)
        # no policy entries for obstacle and terminal states
        index_no_policy = stateObstacles + stateTerminals
        index_policy = [item - 1 for item in range(1, nStates + 1) if item not in index_no_policy]
        # print("index_policy", index_policy)
        # print("index_policy[0]", index_policy[0:2])
        mask = (policy > 0) * mat
        # print("mask", mask)
        # mask = mask.reshape(nRows, nCols, nCols)
        # X3 = X2.reshape(nRows, nCols, nActions)
        # Y3 = Y2.reshape(nRows, nCols, nActions)
        # print("X3 = ", X3)
        # print arrows for policy
        # [N, E, S, W] = [up, right, down, left] = [pi, pi/2, 0, -pi/2]
        alpha = np.pi - np.pi / 2.0 * mask
        # print("alpha", alpha)
        # print("mask ", mask)
        # print("mask test ", np.where(mask[0, :] > 0)[0])
        self._plot_world()
        for i in index_policy:
            # print("ii = ", ii)
            ax = plt.gca()
            # j = int(ii / nRows)
            # i = (ii + 1 - j * nRows) % nCols - 1
            # index = np.where(mask[i, j] > 0)[0]
            index = np.where(mask[i, :] > 0)[0]
            # print("index = ", index)
            # print("X2,Y2", X2[ii, index], Y2[ii, index])
            h = ax.quiver(X2[i, index], Y2[i, index], np.cos(alpha[i, index]), np.sin(alpha[i, index]), color='b')
            # h = ax.quiver(X3[i, j, index], Y3[i, j, index], np.cos(alpha[i, j, index]), np.sin(alpha[i, j, index]),0.3)

        # label states by numbers
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.92, j - 0.92, str(states[k]), fontsize=11, horizontalalignment='right',
                         verticalalignment='bottom')
                k += 1
        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_qvalue(self, Q):
        """
        plot Q-values

        :param Q: matrix of Q-values of size nStates x nActions
        :return: None
        """
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal
        stateObstacles = self.stateObstacles

        fig = plt.plot(1)

        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateHoles + stateObstacles + stateGoal:
                    # print("Q = ", Q)
                    plt.text(i + 0.5, j - 0.15, str(self._truncate(Q[k, 0], 3)), fontsize=8,
                             horizontalalignment='center', verticalalignment='top', multialignment='center')
                    plt.text(i + 0.9, j - 0.5, str(self._truncate(Q[k, 1], 3)), fontsize=8,
                             horizontalalignment='right', verticalalignment='center', multialignment='right')
                    plt.text(i + 0.5, j - 0.85, str(self._truncate(Q[k, 2], 3)), fontsize=8,
                             horizontalalignment='center', verticalalignment='bottom', multialignment='center')
                    plt.text(i + 0.1, j - 0.5, str(self._truncate(Q[k, 3], 3)), fontsize=8,
                             horizontalalignment='left', verticalalignment='center', multialignment='left')
                    # plot cross
                    plt.plot([i, i + 1], [j - 1, j], 'black', lw=0.5)
                    plt.plot([i + 1, i], [j - 1, j], 'black', lw=0.5)
                k += 1

        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def get_nrows(self):

        return self.nRows

    def get_ncols(self):

        return self.nCols

    def get_stateTerminals(self):

        return self.stateTerminals

    def get_stateHoles(self):

        return self.stateHoles

    def get_stateObstacles(self):

        return self.stateObstacles

    def get_stateGoal(self):

        return self.stateGoal

    def get_nstates(self):

        return self.nStates

    def get_nactions(self):

        return self.nActions

    def step(self, action):

        nStates = self.nStates
        stateGoal = self.get_stateGoal()
        stateTerminals = self.get_stateTerminals()

        state = self.observation[0]

        # generate reward and transition model
        p_success = 0.8
        r = -0.04
        self.get_transition_model(p_success)
        self.get_reward_model(r, p_success)
        Pr = self.transition_model
        R = self.reward

        prob = np.array(Pr[state - 1, :, action])
        # print("prob =", prob)
        next_state = np.random.choice(np.arange(1, nStates + 1), p=prob)
        # print("state = ", state)
        # print("next_state inside = ", next_state)
        # print("action = ", action)
        reward = R[state - 1, next_state - 1, action]
        # print("reward = ", R[:, :, 0])
        observation = next_state

        # if (next_state in stateTerminals) or (self.nsteps >= self.max_episode_steps):
        if (next_state in stateTerminals):
            done = True
        else:
            done = False

        self.observation = [next_state]

        return observation, reward, done

    def reset(self, *args):

        nStates = self.nStates

        if not args:
            observation = self.stateInitial
        else:
            observation = []
            while not (observation):
                observation = np.setdiff1d(np.random.choice(np.arange(1, nStates + 1, dtype=int)),
                                           self.stateHoles + self.stateObstacles + self.stateGoal)
        self.observation = observation

    def render(self):

        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        observation = self.observation  # observation
        state = observation[0]

        J, I = np.unravel_index(state - 1, (nRows, nCols), order='F')

        J = (nRows - 1) - J

        circle = plt.Circle((I + 0.5, J + 0.5), 0.28, color='black')
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle)

        self.plot()

    def close(self):
        plt.pause(0.3)  # 0.5
        plt.close()

    def show(self):
        plt.ion()
        plt.show()

    def value_iteration(self, eps=1e-5, synchronous=True, gamma_coeff=0.9):
        values = np.zeros(self.nStates)
        p_success = 0.8
        r = -0.04
        self.get_transition_model(p_success)
        self.get_reward_model(r, p_success)
        Pr = self.transition_model
        R = self.reward
        transition_reward = (Pr * R).sum(axis=1)
        all_values = [values]
        all_actions = []
        delta = 10 + eps
        t = 0
        actionValue = np.zeros((self.nStates, self.nActions))
        while delta > eps:
            t += 1
            delta = 0.0
            next_values = np.zeros(self.nStates)
            curr_action = np.zeros(self.nStates)
            for s in range(self.nStates):
                pr_state = Pr[s, ...]
                sigma = values.dot(pr_state)
                reward_state = transition_reward[s, :] + gamma_coeff * sigma

                actionValue[s, :] = reward_state

                max_value = reward_state.max()
                curr_action[s] = np.argmax(reward_state)
                # update delta - max absolute difference
                delta = max(delta, np.abs(values[s] - max_value))
                if synchronous:
                    next_values[s] = max_value
                else:
                    values[s] = max_value
            if synchronous:
                values = next_values
            all_values.append(values)
            all_actions.append(curr_action)
        print(f'Value iteration ended in {t} iterations with synchronous = {synchronous}')
        return all_values, all_actions, actionValue

    def take_action(self, pi):
        return np.random.choice(range(self.nActions), p=pi[self.observation[0] - 1])

    def random_num_per_grp_cumsumed(self, L):
        # For each element in L pick a random number within range specified by it
        # The final output would be a cumsumed one for use with indexing, etc.
        r1 = np.random.rand(np.sum(L)) + np.repeat(np.arange(len(L)), L)
        offset = np.r_[0, np.cumsum(L[:-1])]
        return r1.argsort()[offset]

    def argmax_per_row_randtie(self, a):
        max_mask = a == a.max(1, keepdims=1)
        m, n = a.shape
        all_argmax_idx = np.flatnonzero(max_mask)
        offset = np.arange(m) * n
        return all_argmax_idx[self.random_num_per_grp_cumsumed(max_mask.sum(1))] - offset

    def epsilon_greedy(self, q, n_visits, eps):
        one_hot_argmax = np.eye(self.nActions)[self.argmax_per_row_randtie(q)]
        pi = one_hot_argmax * (1 - eps(n_visits)) + eps(n_visits) / self.nActions
        return pi

    def gen_episode(self, pi, explore_starts=False):
        done = False
        t = 0
        traj_s = []
        traj_a = []
        traj_r = []
        self.reset(explore_starts)
        traj_s.append(self.observation[0])
        while not done:
            action = self.take_action(pi)  # take a random action
            next_state, reward, done = self.step(action)  # observe next_state and reward
            t += 1
            traj_s.append(next_state)
            traj_a.append(action)
            traj_r.append(reward)
            if done:
                # print("Episode finished after {} timesteps".format(t + 1))
                break
        return traj_s, traj_a, traj_r

    def calculate_return(self, traj_r, gamma=0.9):
        return np.flip(np.flip(gamma ** np.arange(len(traj_r)) * np.array(traj_r)).cumsum()) / (
                gamma ** np.arange(len(traj_r)))

    def mc_control(self, n_episodes, glie=lambda x: 1 / x, lr=1e-4, gamma=0.9, explore_starts=False):
        q = np.zeros((self.nStates, self.nActions))
        n_visits = np.zeros((self.nStates, 1))
        for i in range(n_episodes):
            pi = self.epsilon_greedy(q, n_visits, glie)
            traj_s, traj_a, traj_r = self.gen_episode(pi, explore_starts=explore_starts)
            T = len(traj_r)
            mask_state_action = np.zeros_like(q) == 1
            gt_return = self.calculate_return(traj_r, gamma=gamma)
            for t in range(T):
                state, action = traj_s[t], traj_a[t]
                if not mask_state_action[state - 1, action]:
                    q[state - 1, action] += lr * (gt_return[t] - q[state - 1, action])
                    mask_state_action[state - 1, action] = True
            n_visits[np.any(mask_state_action, axis=-1)] += 1
        return pi, q

    def inside_loop(self, q, eps, lr, loop_type, n_visits, gamma=0.9, explore_starts=True):
        t = 0
        pi = self.epsilon_greedy(q, n_visits, eps)
        self.reset(explore_starts)
        done = False
        action = self.take_action(pi)
        while not done:
            state = self.observation[0]
            n_visits[state-1] += 1
            next_state, reward, done = self.step(action)
            if loop_type == 'sarsa':
                next_action = self.take_action(pi)
            elif loop_type == 'q_learning':
                next_action = np.argmax(q[next_state - 1, :], axis=-1)
            else:
                assert False, f'The inserted loop_type {loop_type} is not supported'
            q[state - 1, action] += lr * (reward + gamma * q[next_state - 1, next_action] - q[state - 1, action])
            pi = self.epsilon_greedy(q, n_visits, eps)
            action = next_action
            t += 1
        return q, n_visits

    def sarsa(self, n_episodes, lr=1.5e-3, glie=lambda x: 1 / x):
        # q = np.random.randn(20, 4)
        q=np.zeros((self.nStates,self.nActions))
        q[np.array(self.stateTerminals) - 1, :] = 0
        n_visits = np.zeros((self.nStates, 1))
        for i in range(n_episodes):
            q, n_visits = self.inside_loop(q, glie, lr, n_visits=n_visits, loop_type='sarsa')

        return q, self.epsilon_greedy(q, 1e20*np.ones((self.nStates, 1)), glie)

    def q_learning(self, n_episodes, lr=1.5e-3, glie=lambda x: 1 / x):

        q = np.zeros((self.nStates, self.nActions))
        q[np.array(self.stateTerminals) - 1, :] = 0
        n_visits = np.zeros((self.nStates, 1))
        for i in range(n_episodes):
            q, n_visits = self.inside_loop(q, glie, lr, n_visits=n_visits, loop_type='q_learning')

        return q, self.epsilon_greedy(q, 1e20*np.ones((self.nStates, 1)), glie)
