def evolved_grad_ar(rep_grads, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):
    """Per-tensor xm.all_reduce in a Python loop (developer baseline)."""
    inv = 1.0 / world_size
    out = []
    for g in rep_grads:
        out.append(xm.all_reduce(xm.REDUCE_SUM, g) * inv)
    return out
