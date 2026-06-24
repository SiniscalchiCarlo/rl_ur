import numpy as np
from collections import defaultdict
from tqdm import tqdm


def legal_indices(legal_moves):
    """
    Convert a list of legal (start, destination) moves into their flat action
    indices. The action map is all_actions = [(i, j) for i in range(16) for j in
    range(16)], so index((i, j)) == i * 16 + j (no dict lookup needed).
    """
    return [start * 16 + dest for (start, dest) in legal_moves]


def e_greedy(Q, state, legal_idx, epsilon=0.1):
    """
    Epsilon-greedy action selection restricted to the legal actions of `state`.

    legal_idx: precomputed list of legal action indices for `state`. Selection is
    masked to these directly (uniform random among legal actions w.p. epsilon,
    otherwise the legal action with the highest Q value, ties broken at random).
    """
    if np.random.random() < epsilon:
        return int(np.random.choice(legal_idx))
    q = Q[state][legal_idx]
    best = np.flatnonzero(q == q.max())
    return int(legal_idx[np.random.choice(best)])


def policy_probs(Q, state, legal_idx, epsilon=0.1):
    """
    Epsilon-greedy probability distribution over the LEGAL actions of `state`.

    Returns an array aligned with `legal_idx`: each legal action gets the
    exploration mass epsilon / n_legal, and the greedy mass (1 - epsilon) is
    split uniformly across the best legal action(s). This is the distribution
    Expected SARSA bootstraps against, so it must match e_greedy (legal-masked,
    random tie-breaking) exactly.
    """
    n = len(legal_idx)
    p = np.full(n, epsilon / n)
    q = Q[state][legal_idx]
    best = np.flatnonzero(q == q.max())
    p[best] += (1.0 - epsilon) / len(best)
    return p


def exp_sarsa(env, num_episodes, discount_factor=1.0, alpha=0.01, epsilon=0.1):
    """
    EXPECTED SARSA algorithm: On-policy TD control. Finds the optimal
    epsilon-greedy policy.

    Args:
        env: Royal Game of Ur environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, episode_rewards, episode_wins, episode_lengths).
        Q maps state -> numpy array of action values (256 flat actions).
    """
    all_actions = [(i, j) for i in range(16) for j in range(16)]
    idx_to_action = {idx: a for idx, a in enumerate(all_actions)}
    action_to_idx = {a: idx for idx, a in enumerate(all_actions)}

    tracked_state = (0, 0, 0, 0, 4)
    tracked_action = action_to_idx[(0, 4)]  
    Q_trajectory = np.zeros(num_episodes)

    Q = defaultdict(lambda: np.zeros(len(all_actions)))  # 256 values

    # Sum of (shaped) rewards per episode; a learning-signal diagnostic only.
    episode_rewards = np.zeros(num_episodes)
    # True win flag per episode (1 = P1 won), set from the SIGN of the terminal
    # reward. Use THIS for the win-ratio plot, not episode_rewards.
    episode_wins = np.zeros(num_episodes)
    episode_lengths = np.zeros(num_episodes)

    for i in tqdm(range(num_episodes)):
        obs, info = env.reset()
        state = tuple(obs)
        legal_idx = legal_indices(env.get_legal_moves())
        action_idx = e_greedy(Q, state, legal_idx, epsilon)

        while True:
            next_obs, reward, done, truncated, info = env.step(idx_to_action[action_idx])
            next_state = tuple(next_obs)
            episode_rewards[i] += reward
            episode_lengths[i] += 1

            if done:
                # No next state to bootstrap from: the target is just the reward.
                target = reward
                episode_wins[i] = 1.0 if reward > 0 else 0.0
            else:
                # Expected value of Q(next_state, .) under the epsilon-greedy
                # policy, restricted to the legal actions of next_state.
                next_legal_idx = legal_indices(info["legal_actions"])
                probs = policy_probs(Q, next_state, next_legal_idx, epsilon)
                expected_q = np.dot(probs, Q[next_state][next_legal_idx])
                target = reward + discount_factor * expected_q

            Q[state][action_idx] += alpha * (target - Q[state][action_idx])

            if done:
                break

            state = next_state
            legal_idx = next_legal_idx
            action_idx = e_greedy(Q, state, legal_idx, epsilon)
        Q_trajectory[i] = Q[tracked_state][tracked_action]

    return Q_trajectory, episode_rewards, episode_wins, episode_lengths
