
def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Build weight tensor by stacking chunks
    chunks = []
    for r in range(W):
        val = 1.0 + 0.5 * (r % 3)
        chunks.append(torch.full((S,), val, device=x.device, dtype=x.dtype))
    a = torch.cat(chunks)
    
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    buf0 = a * s1 / W
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    s2 = s2 / torch.clamp(a, min=1e-9)
    bufN = a * s2 / W
    out = xm.all_reduce(xm.REDUCE_SUM, bufN)
    
    return out
