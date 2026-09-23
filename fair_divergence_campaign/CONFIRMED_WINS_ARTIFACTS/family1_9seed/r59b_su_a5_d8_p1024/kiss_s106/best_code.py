
def r59b_su_a5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    
    # Create weight tensor using torch ops
    rank_ids = torch.arange(W, device=x.device, dtype=torch.long)
    weights = (1.0 + 0.4 * (rank_ids % 5).to(x.dtype)) / W
    a_fwd_t = weights.repeat_interleave(S)
    
    return xm.all_reduce(xm.REDUCE_SUM, xm.all_reduce(xm.REDUCE_SUM, x) * a_fwd_t)
