import random

import numpy as np

from env import ASSIGN_OFFSET, REST, WAIT


def select_action(obs, policy="spt", rest_threshold=0.72):
    mask = obs["mask"]
    if len(mask) == 0:
        return WAIT
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        return WAIT

    features = obs["features"]
    fatigue = float(features[0, 7]) if len(features) else 0.0

    if policy == "random":
        return int(random.choice(legal))

    if policy == "rest_aware" and REST in legal and fatigue >= rest_threshold:
        return REST

    assign = [idx for idx in legal if idx >= ASSIGN_OFFSET]
    if not assign:
        if policy == "rest_aware" and REST in legal:
            return REST
        return WAIT

    if policy in ("spt", "rest_aware"):
        # Candidate feature layout: global(6), worker(4), action(9).
        # action[5] is normalized estimated actual processing time.
        return min(assign, key=lambda idx: features[idx, 15])

    if policy == "fatigue":
        # Prefer jobs that leave lower worker fatigue after completion.
        return min(assign, key=lambda idx: features[idx, 17])

    return min(assign, key=lambda idx: features[idx, 15])


def run_heuristic(env, policy="spt", max_decisions=100000):
    obs = env.reset()
    total_reward = 0.0
    decisions = 0
    while not obs["done"] and decisions < max_decisions:
        action = select_action(obs, policy=policy)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        decisions += 1
    return {
        "policy": policy,
        "reward": total_reward,
        "makespan": env.time,
        "decisions": decisions,
        "force_rest_time": int(env.force_rest_time.sum()),
        "invalid_actions": env.invalid_actions,
        "history": list(env.history),
    }
