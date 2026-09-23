
def r59b_su_a4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Construct weights using torch operations
    rank_idx = torch.arange(W, device=x.device, dtype=torch.long)
    w = (1.0 + 0.6 * (rank_idx % 4)).to(x.dtype) / W
    weights = w.repeat_interleave(S)
    s = s * weights
    return xm.all_reduce(xm.REDUCE_SUM, s)
