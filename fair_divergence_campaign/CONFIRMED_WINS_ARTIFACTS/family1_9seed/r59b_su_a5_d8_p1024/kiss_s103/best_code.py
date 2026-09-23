
def r59b_su_a5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    a = [1.0 + 0.4*(r % 5) for r in range(W)]
    
    # Build weights using cat
    weight_chunks = []
    for r in range(W):
        w = a[r] / W
        chunk = torch.full((S,), w, device=x.device, dtype=x.dtype)
        weight_chunks.append(chunk)
    
    weights = torch.cat(weight_chunks)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = W * weights * s
    
    return s
