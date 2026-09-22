
def r21_su8_countonly_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Create weight tensor using vectorized operations
    indices = torch.arange(W*S, device=x.device, dtype=torch.long)
    rank_indices = indices // S
    weights_scaled = (1.0 + 0.5*(rank_indices % 3).to(x.dtype)) / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = xm.all_reduce(xm.REDUCE_SUM, weights_scaled * s)
    
    return s
