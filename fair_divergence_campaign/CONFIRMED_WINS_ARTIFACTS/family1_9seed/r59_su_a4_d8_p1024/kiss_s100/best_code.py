
def r59_su_a4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    
    # Create weights using repeat
    a = [1.0 + 0.6*(r % 4) for r in range(world_size)]
    w = torch.tensor([[a[r] / world_size] for r in range(world_size)], device=x.device, dtype=x.dtype)
    weights = w.repeat(1, S).view(-1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    return xm.all_reduce(xm.REDUCE_SUM, s * weights)
