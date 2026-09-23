
def r63_gmean_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # After first all_reduce, s is same on all ranks
    # So all_reduce_sum(s + s.mean()) = world_size * (s + s.mean())
    # Dividing by world_size gives back s + s.mean()
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    return s + s.mean()
