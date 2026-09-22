
def r26_perm_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    perm = [(r + W // 2) % W for r in range(W)]
    inv_perm = [0] * W
    for r in range(W):
        inv_perm[perm[r]] = r
    
    fwd_scale = [a[inv_perm[p]] / W for p in range(W)]
    
    fwd_t = torch.tensor(fwd_scale, device=x.device, dtype=x.dtype).view(W, 1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(W, S)
    
    return xm.all_reduce(xm.REDUCE_SUM, (fwd_t * s).view(-1))
