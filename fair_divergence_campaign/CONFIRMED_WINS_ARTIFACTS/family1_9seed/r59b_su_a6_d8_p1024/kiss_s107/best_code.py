
def r59b_su_a6_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    
    # Create weights using torch operations
    r_vals = torch.arange(W, device=x.device, dtype=x.dtype)
    a_tensor = 1.0 + 0.35 * (r_vals % 6)
    weights_fwd_per_rank = a_tensor / float(W)
    weights_fwd = weights_fwd_per_rank.unsqueeze(1).repeat(1, S).reshape(-1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = xm.all_reduce(xm.REDUCE_SUM, weights_fwd * s)
    
    return s
