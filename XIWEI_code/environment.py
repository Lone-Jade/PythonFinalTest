"""
Event-driven JSP Scheduling Environment with Worker Fatigue.

Models a Flexible Job Shop Scheduling Problem where:
- Each job has sequential operations
- Each operation requires a specific machine
- Any worker can operate any machine (with different skill levels)
- Worker fatigue accumulates during work and recovers during idle time
- Fatigue impacts processing efficiency

Reward design: immediate reward = -(assigned operation's actual processing time).
This gives direct credit assignment — each action's reward reflects its own
consequence, not unrelated concurrent operations. Terminal fatigue penalty is
added when the episode ends.
"""

from typing import Dict, Tuple, List
import numpy as np

from config import EnvConfig


class JSPEnvironment:
    """
    Job Shop Scheduling Environment with Personnel and Fatigue.

    State space (flattened vector, normalized to ~[0,1]):
        [machine_free(M), machine_remaining_time_norm(M),
         worker_free(W), worker_fatigue_norm(W),
         job_progress(N), job_next_machine_norm(N),
         current_time_norm(1)]

    Action space: (job_id, worker_id) pairs, dim = N * W

    One step() = one scheduling decision -> advance to next decision point.
    Reward = -(assigned operation time) / reward_scale + terminal penalty.
    """

    def __init__(self, data: Dict, env_config: EnvConfig = None):
        self.data = data
        self.cfg = env_config or EnvConfig()

        self.num_jobs = data['num_jobs']
        self.num_machines = data['num_machines']
        self.num_workers = data['num_workers']
        self.machine_ops = data['machine_ops']            # (J, ops) -> machine_id (1-indexed)
        self.base_times = data['processing_times']        # (J, ops, W) -> time

        self.state_dim = (3 * self.num_machines + 3 * self.num_workers +
                          3 * self.num_jobs + 1)
        self.action_dim = self.num_jobs * self.num_workers

        # Reward normalization
        self.max_time_estimate = max(np.sum(np.median(self.base_times, axis=-1)), 1.0)
        self.reward_scale = self.max_time_estimate / max(self.num_machines, 1)

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state. Returns initial state vector."""
        self.machine_busy = np.zeros(self.num_machines, dtype=bool)
        self.machine_remaining = np.zeros(self.num_machines, dtype=np.float32)

        self.worker_busy = np.zeros(self.num_workers, dtype=bool)
        self.worker_fatigue = np.zeros(self.num_workers, dtype=np.float32)

        self.job_next_op = np.zeros(self.num_jobs, dtype=np.int32)
        self.job_complete = np.zeros(self.num_jobs, dtype=bool)
        self.job_total_ops = self.num_machines

        self.current_time = 0.0

        self.job_start_times = np.full(self.num_jobs, -1.0, dtype=np.float32)
        self.job_completion_times = np.full(self.num_jobs, -1.0, dtype=np.float32)
        self.worker_total_work_time = np.zeros(self.num_workers, dtype=np.float32)
        self.worker_idle_time = np.zeros(self.num_workers, dtype=np.float32)

        self.active_ops: List[Tuple[float, int, int, int]] = []

        self.total_reward = 0.0
        self.step_count = 0

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Build the normalized state vector (all components in [0,1]).

        State = [machine_free(M), machine_remaining_norm(M),
                 worker_free(W), worker_fatigue_norm(W),
                 job_progress(N), job_next_machine_norm(N),
                 job_remaining_work_norm(N),       # NEW: critical path signal
                 machine_demand_norm(M),           # NEW: bottleneck signal
                 worker_load_norm(W),              # NEW: workload balance signal
                 current_time_norm(1)]
        Total dims = 3M + 3W + 3N + 1
        """
        state = []

        # --- Machine features (2M) ---
        for m in range(self.num_machines):
            state.append(1.0 if not self.machine_busy[m] else 0.0)
            state.append(min(self.machine_remaining[m] / (self.max_time_estimate + 1.0), 1.0))

        # --- Worker features (2W) ---
        for w in range(self.num_workers):
            state.append(1.0 if not self.worker_busy[w] else 0.0)
            state.append(min(self.worker_fatigue[w] / 5.0, 1.0))

        # --- Job features (2N + N) ---
        for j in range(self.num_jobs):
            if self.job_complete[j]:
                state.append(1.0)       # progress = 100%
                state.append(0.0)       # next machine = none
                state.append(0.0)       # remaining work = 0
            else:
                state.append(self.job_next_op[j] / self.job_total_ops)
                machine_id = self.machine_ops[j, self.job_next_op[j]]
                state.append(machine_id / (self.num_machines + 1))
                # Remaining work: sum of median base times for unfinished ops
                rem_work = 0.0
                for op in range(self.job_next_op[j], self.job_total_ops):
                    rem_work += float(np.median(self.base_times[j, op, :]))
                state.append(min(rem_work / (self.max_time_estimate + 1.0), 1.0))

        # --- Machine demand (M): jobs waiting for each machine ---
        machine_demand = np.zeros(self.num_machines, dtype=np.float32)
        for j in range(self.num_jobs):
            if not self.job_complete[j]:
                needed_m = self.machine_ops[j, self.job_next_op[j]] - 1
                machine_demand[needed_m] += 1
        for m in range(self.num_machines):
            state.append(machine_demand[m] / max(self.num_jobs, 1))

        # --- Worker cumulative load (W): fraction of time spent working ---
        for w in range(self.num_workers):
            load = (self.worker_total_work_time[w] /
                    max(self.current_time, 1.0))
            state.append(min(load, 1.0))

        state.append(min(self.current_time / (self.max_time_estimate * 2.0 + 1.0), 1.0))

        return np.array(state, dtype=np.float32)

    def _get_action_mask(self) -> np.ndarray:
        """Returns boolean mask of shape (action_dim,). True = valid action."""
        mask = np.zeros(self.action_dim, dtype=bool)

        for j in range(self.num_jobs):
            if self.job_complete[j]:
                continue
            needed_machine = self.machine_ops[j, self.job_next_op[j]] - 1
            if self.machine_busy[needed_machine]:
                continue
            for w in range(self.num_workers):
                if not self.worker_busy[w]:
                    mask[j * self.num_workers + w] = True

        return mask

    def get_valid_actions(self) -> np.ndarray:
        """Return indices of valid actions."""
        return np.where(self._get_action_mask())[0]

    def _advance_fatigue(self, delta_t: float):
        """Update fatigue for all workers over a time interval."""
        for w in range(self.num_workers):
            if self.worker_busy[w]:
                self.worker_fatigue[w] += self.cfg.alpha * delta_t
                self.worker_total_work_time[w] += delta_t
            else:
                self.worker_fatigue[w] = max(
                    0.0, self.worker_fatigue[w] - self.cfg.beta * delta_t
                )
                self.worker_idle_time[w] += delta_t
            # Cap fatigue to prevent runaway feedback loop
            if self.worker_fatigue[w] > self.cfg.F_max:
                self.worker_fatigue[w] = self.cfg.F_max

    def _process_completions(self):
        """Process all operations that have completed at or before current_time."""
        completed = [op for op in self.active_ops if op[0] <= self.current_time]
        self.active_ops = [op for op in self.active_ops if op[0] > self.current_time]

        for _, m_id, w_id, j_id in completed:
            self.machine_busy[m_id] = False
            self.machine_remaining[m_id] = 0.0
            self.worker_busy[w_id] = False
            self.job_next_op[j_id] += 1
            if self.job_next_op[j_id] >= self.job_total_ops:
                self.job_complete[j_id] = True
                self.job_completion_times[j_id] = self.current_time

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one scheduling decision and advance to the NEXT decision point.

        Reward design (action-specific for proper credit assignment):
        - Immediate: -(assigned operation's actual_time) / reward_scale
        - Terminal: fatigue penalty (only when episode ends)

        Args:
            action: integer = job_id * num_workers + worker_id

        Returns:
            state: new state vector at next decision point (or terminal)
            reward: normalized reward for THIS action
            done: whether episode is complete
            info: dict with diagnostic information
        """
        job_id = action // self.num_workers
        worker_id = action % self.num_workers

        # --- 1. Validate ---
        mask = self._get_action_mask()
        if not mask[action]:
            return self._get_state(), -10.0, False, {'invalid': True}

        # --- 2. Assign operation ---
        op_idx = self.job_next_op[job_id]
        machine_id = self.machine_ops[job_id, op_idx] - 1

        base_time = self.base_times[job_id, op_idx, worker_id]
        fatigue = self.worker_fatigue[worker_id]
        actual_time = base_time * (1.0 + self.cfg.gamma * fatigue)

        if self.job_start_times[job_id] < 0:
            self.job_start_times[job_id] = self.current_time

        self.machine_busy[machine_id] = True
        self.machine_remaining[machine_id] = actual_time
        self.worker_busy[worker_id] = True
        self.active_ops.append(
            (self.current_time + actual_time, machine_id, worker_id, job_id)
        )

        # --- 3. Reward: directly tied to this action's consequence ---
        reward = -actual_time / self.reward_scale

        # --- 4. Advance through events until next decision point ---
        done = False

        while len(self.active_ops) > 0:
            next_event = min(op[0] for op in self.active_ops)
            delta_t = next_event - self.current_time

            if delta_t > 0:
                self._advance_fatigue(delta_t)

            self.current_time = next_event
            self._process_completions()

            # Check termination
            if np.all(self.job_complete):
                done = True
                # Terminal makespan reward (direct makespan signal)
                if self.cfg.use_terminal_ms_reward:
                    reward += -self.current_time / self.reward_scale
                # Fatigue penalty
                fatigue_excess = np.sum(
                    np.maximum(0, self.worker_fatigue - self.cfg.F_threshold)
                )
                reward += -self.cfg.lambda_fatigue * fatigue_excess / max(self.num_workers, 1)
                break

            # Stop at next decision point
            if np.any(self._get_action_mask()):
                break

        self.total_reward += reward
        self.step_count += 1

        info = {
            'invalid': False,
            'makespan': self.current_time if done else None,
        }

        return self._get_state(), reward, done, info

    def get_makespan(self) -> float:
        """Return the final makespan (max completion time)."""
        if np.all(self.job_complete):
            return self.current_time
        return float('inf')

    def get_avg_fatigue(self) -> float:
        """Return average worker fatigue."""
        return float(np.mean(self.worker_fatigue))


class GreedyScheduler:
    """Baseline: greedy heuristic that picks the shortest processing time."""

    def __init__(self, data: Dict, env_config: EnvConfig = None):
        self.data = data
        self.env = JSPEnvironment(data, env_config)

    def solve(self) -> Tuple[float, float, List]:
        """Run greedy scheduling. Returns (makespan, avg_fatigue, schedule)."""
        self.env.reset()
        schedule = []

        while not np.all(self.env.job_complete):
            valid_actions = self.env.get_valid_actions()
            if len(valid_actions) == 0:
                break

            best_action = valid_actions[0]
            best_time = float('inf')
            for a in valid_actions:
                j_id = a // self.env.num_workers
                w_id = a % self.env.num_workers
                op_idx = self.env.job_next_op[j_id]
                base_t = self.env.data['processing_times'][j_id, op_idx, w_id]
                fatigue = self.env.worker_fatigue[w_id]
                actual_t = base_t * (1.0 + self.env.cfg.gamma * fatigue)
                if actual_t < best_time:
                    best_time = actual_t
                    best_action = a

            _, _, done, _ = self.env.step(int(best_action))
            schedule.append({
                'job': int(best_action) // self.env.num_workers,
                'worker': int(best_action) % self.env.num_workers,
                'time': self.env.current_time,
            })

            if done:
                break

        return self.env.get_makespan(), self.env.get_avg_fatigue(), schedule
