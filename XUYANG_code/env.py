import math
from dataclasses import dataclass

import numpy as np

from config import EnvConfig


WAIT = 0
REST = 1
ASSIGN_OFFSET = 2


@dataclass
class ScheduledTask:
    job: int
    op: int
    machine: int
    worker: int
    start: int
    finish: int
    base_time: float
    actual_time: int
    fatigue_before: float
    fatigue_after: float


def exp_unit(x, k):
    x = float(np.clip(x, 0.0, 1.0))
    return (math.exp(k * x) - 1.0) / (math.exp(k) - 1.0)


def log_unit(x, k):
    x = float(np.clip(x, 0.0, 1.0))
    return math.log1p(k * x) / math.log1p(k)


class JobShopFatigueEnv:
    """Event-driven SMDP environment for worker-aware job-shop scheduling."""

    def __init__(self, instance, config=None, t_ref=None, seed=0):
        self.instance = instance
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.t_ref = max(float(t_ref or self._default_t_ref()), self.config.min_t_ref)
        self.reset()

    def _default_t_ref(self):
        best_worker_time = self.instance.processing_times.min(axis=2).sum()
        return best_worker_time / max(1, self.instance.n_machines)

    def reset(self):
        inst = self.instance
        self.time = 0
        self.done = False
        self.job_next_op = np.zeros(inst.n_jobs, dtype=np.int64)
        self.job_ready = np.zeros(inst.n_jobs, dtype=np.int64)
        self.machine_busy_until = np.zeros(inst.n_machines, dtype=np.int64)
        self.machine_task = np.full(inst.n_machines, -1, dtype=np.int64)
        self.worker_busy_until = np.zeros(inst.n_workers, dtype=np.int64)
        self.worker_status = np.array(["idle"] * inst.n_workers, dtype=object)
        self.worker_task = np.full(inst.n_workers, -1, dtype=np.int64)
        self.fatigue = np.zeros(inst.n_workers, dtype=np.float32)
        self.force_rest_time = np.zeros(inst.n_workers, dtype=np.int64)
        self.history = []
        self.decision_worker = None
        self.decided_workers = set()
        self.invalid_actions = 0
        self.last_interval_cost = 0.0
        self._advance_to_decision()
        return self.observe()

    def all_done(self):
        all_ops_scheduled = bool(np.all(self.job_next_op >= self.instance.n_machines))
        no_working = not any(status == "working" for status in self.worker_status)
        return all_ops_scheduled and no_working

    def _sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def processing_multiplier(self, fatigue):
        cfg = self.config
        return 1.0 + cfg.beta * self._sigmoid(cfg.sigmoid_k * (float(fatigue) - cfg.theta))

    def actual_processing_time(self, job, worker):
        op = self.job_next_op[job]
        base = float(self.instance.processing_times[job, op, worker])
        return max(1, int(math.ceil(base * self.processing_multiplier(self.fatigue[worker]))))

    def _work_fatigue_after(self, fatigue, duration):
        return 1.0 - (1.0 - float(fatigue)) * math.exp(-self.config.alpha * duration)

    def _rest_fatigue_after(self, fatigue, duration):
        return float(fatigue) * math.exp(-self.config.gamma_rest * duration)

    def _rest_duration_to_resume(self, fatigue):
        if fatigue <= self.config.f_resume:
            return 0
        return max(1, int(math.ceil(math.log(self.config.f_resume / fatigue) / (-self.config.gamma_rest))))

    def feasible_jobs(self, worker):
        if self.worker_status[worker] != "idle":
            return []
        jobs = []
        for job in range(self.instance.n_jobs):
            op = self.job_next_op[job]
            if op >= self.instance.n_machines:
                continue
            machine = self.instance.machines[job, op]
            if self.job_ready[job] <= self.time and self.machine_busy_until[machine] <= self.time:
                jobs.append(job)
        return jobs

    def _finish_events_at_current_time(self):
        for worker in range(self.instance.n_workers):
            status = self.worker_status[worker]
            if status == "working" and self.worker_busy_until[worker] <= self.time:
                job = int(self.worker_task[worker])
                op = int(self.job_next_op[job] - 1)
                machine = int(self.instance.machines[job, op])
                self.worker_status[worker] = "idle"
                self.worker_task[worker] = -1
                self.machine_task[machine] = -1
                if self.fatigue[worker] >= self.config.f_force:
                    duration = self._rest_duration_to_resume(float(self.fatigue[worker]))
                    self.worker_status[worker] = "force_rest"
                    self.worker_busy_until[worker] = self.time + duration
            elif status in ("force_rest", "active_rest") and self.worker_busy_until[worker] <= self.time:
                self.fatigue[worker] = self._rest_fatigue_after(
                    float(self.fatigue[worker]), max(0, int(self.worker_busy_until[worker] - self.time))
                )
                self.worker_status[worker] = "idle"
                self.worker_task[worker] = -1

    def _candidate_decision_worker(self):
        for worker in range(self.instance.n_workers):
            if worker in self.decided_workers:
                continue
            if self.worker_status[worker] == "idle":
                return worker
        return None

    def _interval_cost(self, delta):
        if delta <= 0:
            return 0.0
        cfg = self.config
        ratio = delta / self.t_ref
        mean_f = float(np.mean(self.fatigue))
        max_f = float(np.max(self.fatigue))

        machine_idle = self._machine_idle_ratio()
        worker_idle = self._worker_idle_ratio()
        force_ratio = float(np.mean(self.worker_status == "force_rest"))
        return (
            cfg.s_time * ratio
            + cfg.s_avg_fatigue * ratio * exp_unit(mean_f, cfg.k_avg_fatigue)
            + cfg.s_max_fatigue * ratio * exp_unit(max_f, cfg.k_max_fatigue)
            + cfg.s_machine_idle * ratio * exp_unit(machine_idle, cfg.k_machine_idle)
            + cfg.s_worker_idle * ratio * log_unit(worker_idle, cfg.k_worker_idle)
            + cfg.s_force_rest * ratio * force_ratio
        )

    def _machine_idle_ratio(self):
        need_machines = set()
        for job in range(self.instance.n_jobs):
            op = self.job_next_op[job]
            if op >= self.instance.n_machines or self.job_ready[job] > self.time:
                continue
            machine = int(self.instance.machines[job, op])
            need_machines.add(machine)
        if not need_machines:
            return 0.0
        idle_needed = sum(1 for m in need_machines if self.machine_busy_until[m] <= self.time)
        return idle_needed / max(1, self.instance.n_machines)

    def _worker_idle_ratio(self):
        idle = sum(1 for s in self.worker_status if s == "idle")
        return idle / max(1, self.instance.n_workers)

    def _advance_time(self):
        future = []
        for t in self.worker_busy_until:
            if t > self.time:
                future.append(int(t))
        for t in self.machine_busy_until:
            if t > self.time:
                future.append(int(t))
        for t in self.job_ready:
            if t > self.time:
                future.append(int(t))
        if not future:
            return 0.0

        next_time = min(future)
        delta = next_time - self.time
        cost = self._interval_cost(delta)

        for worker in range(self.instance.n_workers):
            status = self.worker_status[worker]
            if status == "force_rest":
                self.force_rest_time[worker] += delta
            if status in ("force_rest", "active_rest"):
                self.fatigue[worker] = self._rest_fatigue_after(self.fatigue[worker], delta)

        self.time = next_time
        self.decided_workers.clear()
        self._finish_events_at_current_time()
        return cost

    def _advance_to_decision(self):
        total_cost = 0.0
        while not self.all_done():
            self._finish_events_at_current_time()
            worker = self._candidate_decision_worker()
            if worker is not None:
                self.decision_worker = worker
                self.last_interval_cost = total_cost
                return
            cost = self._advance_time()
            total_cost += cost
            if cost == 0.0 and self._candidate_decision_worker() is None:
                self.decided_workers.clear()
                worker = self._candidate_decision_worker()
                if worker is not None:
                    self.decision_worker = worker
                    self.last_interval_cost = total_cost
                    return
                break
        self.done = self.all_done()
        self.decision_worker = None
        self.last_interval_cost = total_cost

    def _global_features(self):
        remaining = np.sum(self.job_next_op < self.instance.n_machines)
        # Soft-clip to max 50 to keep within reasonable range for NN
        time_ratio = min(self.time / max(1.0, self.t_ref), 50.0)
        return np.array(
            [
                time_ratio,
                remaining / max(1, self.instance.n_jobs),
                np.mean(self.fatigue),
                np.max(self.fatigue),
                self._machine_idle_ratio(),
                self._worker_idle_ratio(),
            ],
            dtype=np.float32,
        )

    def candidate_features(self):
        """Return action names, feature matrix and legal mask for current worker."""
        if self.done or self.decision_worker is None:
            return [], np.zeros((0, self.feature_dim), dtype=np.float32), np.zeros(0, dtype=bool)

        worker = self.decision_worker
        global_f = self._global_features()
        worker_f = np.array(
            [
                worker / max(1, self.instance.n_workers - 1),
                float(self.fatigue[worker]),
                1.0 if self.worker_status[worker] == "idle" else 0.0,
                1.0 if self.fatigue[worker] >= self.config.f_force else 0.0,
            ],
            dtype=np.float32,
        )

        names = ["WAIT", "REST"]
        feats = []
        masks = []

        feats.append(np.concatenate([global_f, worker_f, np.zeros(9, dtype=np.float32)]))
        masks.append(True)

        rest_need = max(0.0, self.fatigue[worker] - self.config.f_resume) / max(1e-6, 1 - self.config.f_resume)
        feats.append(np.concatenate([global_f, worker_f, np.array([1, rest_need, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)]))
        masks.append(True)

        feasible = set(self.feasible_jobs(worker))
        for job in range(self.instance.n_jobs):
            op = self.job_next_op[job]
            if op >= self.instance.n_machines:
                machine = 0
                base = 0.0
                actual = 0.0
                legal = False
            else:
                machine = int(self.instance.machines[job, op])
                base = float(self.instance.processing_times[job, op, worker])
                actual = float(max(1, math.ceil(base * self.processing_multiplier(self.fatigue[worker]))))
                legal = job in feasible and self.fatigue[worker] < self.config.f_force
            remaining_ops = max(0, self.instance.n_machines - int(op))
            fatigue_after = self._work_fatigue_after(self.fatigue[worker], actual) if actual > 0 else self.fatigue[worker]
            # Soft-clip to max 50 for NN stability while preserving magnitude ordering
            base_norm = min(base / max(1.0, self.t_ref), 50.0)
            actual_norm = min(actual / max(1.0, self.t_ref), 50.0)
            action_f = np.array(
                [
                    0.0,
                    job / max(1, self.instance.n_jobs - 1),
                    int(op) / max(1, self.instance.n_machines),
                    machine / max(1, self.instance.n_machines - 1),
                    base_norm,
                    actual_norm,
                    remaining_ops / max(1, self.instance.n_machines),
                    fatigue_after,
                    1.0 if legal else 0.0,
                ],
                dtype=np.float32,
            )
            names.append(f"ASSIGN_J{job + 1}")
            feats.append(np.concatenate([global_f, worker_f, action_f]))
            masks.append(legal)

        return names, np.vstack(feats).astype(np.float32), np.array(masks, dtype=bool)

    @property
    def feature_dim(self):
        return 6 + 4 + 9

    def observe(self):
        names, features, mask = self.candidate_features()
        return {
            "time": self.time,
            "worker": self.decision_worker,
            "action_names": names,
            "features": features,
            "mask": mask,
            "done": self.done,
        }

    def step(self, action_index):
        if self.done:
            return self.observe(), 0.0, True, {"makespan": self.time}

        names, _, mask = self.candidate_features()
        worker = self.decision_worker
        invalid = False
        reward = -self.last_interval_cost

        if action_index < 0 or action_index >= len(names) or not mask[action_index]:
            action_index = WAIT
            invalid = True
            self.invalid_actions += 1
            reward -= self.config.s_invalid

        if action_index == WAIT:
            self.decided_workers.add(worker)
        elif action_index == REST:
            duration = self.config.active_rest_duration
            self.worker_status[worker] = "active_rest"
            self.worker_busy_until[worker] = self.time + duration
            self.worker_task[worker] = -1
            self.decided_workers.add(worker)
        else:
            job = action_index - ASSIGN_OFFSET
            op = int(self.job_next_op[job])
            machine = int(self.instance.machines[job, op])
            base = float(self.instance.processing_times[job, op, worker])
            fatigue_before = float(self.fatigue[worker])
            actual = self.actual_processing_time(job, worker)
            finish = self.time + actual
            fatigue_after = self._work_fatigue_after(fatigue_before, actual)

            self.worker_status[worker] = "working"
            self.worker_busy_until[worker] = finish
            self.worker_task[worker] = job
            self.machine_busy_until[machine] = finish
            self.machine_task[machine] = job
            self.job_next_op[job] += 1
            self.job_ready[job] = finish
            self.fatigue[worker] = fatigue_after
            self.decided_workers.add(worker)
            self.history.append(
                ScheduledTask(job, op, machine, worker, self.time, finish, base, actual, fatigue_before, fatigue_after)
            )

        self._advance_to_decision()
        done = self.done
        if done:
            reward -= self.config.s_makespan * exp_unit(min(self.time / self.t_ref, 3.0) / 3.0, self.config.k_makespan)
        return self.observe(), float(reward), done, {"invalid": invalid, "makespan": self.time}

    def legal_action_count(self):
        obs = self.observe()
        return int(np.sum(obs["mask"]))
