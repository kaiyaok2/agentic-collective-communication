
def perrow_mixed_bigM_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Perform all_reduce on entire tensor instead of per-row
    mx = xm.all_reduce(xm.REDUCE_MAX, x)
    mn = xm.all_reduce(xm.REDUCE_MIN, x)
    return mx - mn
