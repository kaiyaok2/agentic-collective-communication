
def r59b_su_a4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Build weight tensors more efficiently using repeat
    w_per_rank = [1.0 + 0.6*(r % 4) for r in range(W)]
    w_scaled = [w / W for w in w_per_rank]
    inv_w = [1.0 / w for w in w_per_rank]
    
    # Convert to tensors and repeat
    wt = torch.tensor(w_scaled, device=x.device, dtype=x.dtype).repeat_interleave(S)
    iwt = torch.tensor(inv_w, device=x.device, dtype=x.dtype).repeat_interleave(S)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, wt * s)
        s = s * iwt
    
    s = xm.all_reduce(xm.REDUCE_SUM, wt * s)
    
    return s
