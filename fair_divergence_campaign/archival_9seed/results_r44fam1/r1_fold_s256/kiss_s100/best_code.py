
def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    W_inv = 1.0 / W
    
    # Build scaled vector using cat
    chunks = []
    for r in range(W):
        val = (1.0 + 0.5*(r % 3)) * W_inv
        chunk = torch.full((S,), val, device=x.device, dtype=x.dtype)
        chunks.append(chunk)
    a_vec = torch.cat(chunks)
    
    # Chained operations
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    s2 = xm.all_reduce(xm.REDUCE_SUM, a_vec * s1)
    out = xm.all_reduce(xm.REDUCE_SUM, s2 * W_inv)
    
    return out
