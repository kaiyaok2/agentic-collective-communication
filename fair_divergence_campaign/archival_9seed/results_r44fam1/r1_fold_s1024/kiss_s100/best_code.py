def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    
    # Pre-compute scaled factors
    inv_W = 1.0 / W
    a_vals = [(1.0 + 0.5*(r % 3)) * inv_W for r in range(W)]
    a_flat = [a_vals[r] for r in range(W) for _ in range(S)]
    a_tensor = torch.tensor(a_flat, device=x.device, dtype=x.dtype)
    
    # Vectorized operations
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    buf0 = a_tensor * s1
    s1 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    out = xm.all_reduce(xm.REDUCE_SUM, s1 * inv_W)
    return out