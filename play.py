import time

from enviroment import RoyalGameOfUr


def board_positions(positions):
    return [position for position in positions if 1 <= position <= 14]


def describe_action(env, action):
    if action == env.pass_action:
        return "pass"
    return f"move from {env.actions[action]}"


def play_episode(N=2, seed=None, delay=0.5, max_steps=200):
    env = RoyalGameOfUr(N)
    observation, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    print(f"Initial observation: {observation.tolist()}")
    env.render(board_positions(env.player1_loc), board_positions(env.player2_loc))

    while not (terminated or truncated) and step_count < max_steps:
        legal_actions = env.get_legal_moves()
        action = int(env.np_random.choice(legal_actions))

        print(
            f"Step {step_count + 1}: roll={env.roll}, "
            f"legal={legal_actions}, action={action} ({describe_action(env, action)})"
        )

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        print(
            f"Observation: {observation.tolist()}, reward={reward}, "
            f"terminated={terminated}, next_legal={info['legal_actions']}"
        )
        env.render(board_positions(env.player1_loc), board_positions(env.player2_loc))
        time.sleep(delay)

    if step_count >= max_steps and not terminated:
        truncated = True

    winner = "player 1" if reward == 1 else "player 2" if terminated else "none"
    print(
        f"Finished after {step_count} steps. "
        f"winner={winner}, total_reward={total_reward}, truncated={truncated}"
    )


if __name__ == "__main__":
    play_episode(N=2, seed=42)
