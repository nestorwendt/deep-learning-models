"""
"""

import os


def calculate_num_workers() -> int:
    """
    Calculate the number of workers based on the CPU count.

    Returns:
        int: Number of workers.
    """
    num_workers = os.cpu_count()
    num_workers = num_workers if num_workers is not None else 0

    return num_workers
