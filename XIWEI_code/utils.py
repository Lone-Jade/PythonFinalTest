"""
Data loading and utility functions for JSP scheduling.
"""

import os
import csv
import re
import ast
from typing import Dict, List, Tuple

import numpy as np
import torch


def parse_time_list(s: str) -> List[int]:
    """Parse a string like '[41,53]' or '[82,87,79]' into a list of ints."""
    return ast.literal_eval(s)


def load_csv_data(filepath: str) -> Dict:
    """
    Load a JSP CSV file and return structured data.

    Uses csv.reader to properly handle quoted fields containing commas.

    Args:
        filepath: Path to the CSV file.

    Returns:
        dict with keys:
            - num_jobs: int
            - num_machines: int
            - num_workers: int
            - machine_ops: np.ndarray (num_jobs, num_machines) - machine assignment per operation
            - processing_times: np.ndarray (num_jobs, num_machines, num_workers)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    header = all_rows[0]
    # Count machine columns
    machine_cols = [c for c in header if c.startswith('Machine_Op')]
    time_cols = [c for c in header if c.startswith('Time_Op')]
    num_machines = len(machine_cols)

    data_rows = [row for row in all_rows[1:] if row and any(cell.strip() for cell in row)]
    num_jobs = len(data_rows)

    # Parse first time entry to determine num_workers
    first_time = parse_time_list(data_rows[0][1 + num_machines])
    num_workers = len(first_time)

    # Build arrays
    machine_ops = np.zeros((num_jobs, num_machines), dtype=np.int32)
    processing_times = np.zeros((num_jobs, num_machines, num_workers), dtype=np.float32)

    for i, row in enumerate(data_rows):
        for op_idx in range(num_machines):
            machine_ops[i, op_idx] = int(row[1 + op_idx])  # machine IDs are 1-indexed
            times = parse_time_list(row[1 + num_machines + op_idx])
            for w_idx, t in enumerate(times):
                processing_times[i, op_idx, w_idx] = float(t)

    return {
        'num_jobs': num_jobs,
        'num_machines': num_machines,
        'num_workers': num_workers,
        'machine_ops': machine_ops,       # (J, M) -> machine_id (1-indexed)
        'processing_times': processing_times,  # (J, M, W) -> time
    }


def find_csv_files(data_dir: str) -> List[str]:
    """Find all CSV files in the data directory, sorted by size."""
    csv_dir = os.path.join(data_dir, 'csv_output')
    if not os.path.exists(csv_dir):
        return []
    files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    def sort_key(name):
        parts = name.replace('.csv', '').split('x')
        if len(parts) == 3:
            return int(parts[0]) * int(parts[1]) * int(parts[2])
        return 0

    files.sort(key=sort_key)
    return [os.path.join(csv_dir, f) for f in files]


def get_state_dim(data: Dict) -> int:
    """Calculate the state vector dimension.

    State = 3M (machine) + 3W (worker) + 3N (job) + 1 (time)
    """
    N, M, W = data['num_jobs'], data['num_machines'], data['num_workers']
    return 3 * M + 3 * W + 3 * N + 1


def get_action_dim(data: Dict) -> int:
    """Calculate the action space dimension (job × worker)."""
    return data['num_jobs'] * data['num_workers']


def get_data_path(data_dir: str, filename: str) -> str:
    """Get full path to a CSV data file."""
    if not filename.endswith('.csv'):
        filename += '.csv'
    return os.path.join(data_dir, 'csv_output', filename)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: str, device: str = None):
    """Load a model checkpoint with optional metadata.

    Args:
        path: Path to the .pt checkpoint file.
        device: 'cpu', 'cuda', or None (auto-detect).

    Returns:
        Tuple of (checkpoint_dict, metadata_dict_or_None).
        checkpoint_dict has keys: q_network, target_network, optimizer,
                                  steps_done, episodes_done, metadata(optional)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = checkpoint.get('metadata', None)
    return checkpoint, metadata
