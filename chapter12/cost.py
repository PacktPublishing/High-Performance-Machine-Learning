"""Cost per completed task (Code 12.1)."""


def cost_per_task(gpu_price_h, n_gpus, tasks_per_s, utilization=0.6, other_cost_h=0.0):
    """tasks_per_s: goodput, the task rate at which the objectives hold.
    utilization: average load divided by goodput over the billing period."""
    hourly = gpu_price_h * n_gpus + other_cost_h
    tasks_per_h = tasks_per_s * 3600 * utilization
    return hourly / tasks_per_h


if __name__ == "__main__":
    print(f"{cost_per_task(3.95, 1, 0.5, utilization=1.0):.4f}")   # 0.0022
    print(f"{cost_per_task(3.95, 1, 0.5, utilization=0.4):.4f}")   # 0.0055
