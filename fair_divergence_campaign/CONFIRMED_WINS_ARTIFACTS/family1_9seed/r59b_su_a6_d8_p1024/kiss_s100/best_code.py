
def r59b_su_a6_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    a = [1.0 + 0.35*(r % 6) for r in range(W)]
    
    a_scaled_vals = [a[r] / W for r in range(W) for _ in range(S)]
    a_scaled_tensor = torch.tensor(a_scaled_vals, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = a_scaled_tensor * s
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
