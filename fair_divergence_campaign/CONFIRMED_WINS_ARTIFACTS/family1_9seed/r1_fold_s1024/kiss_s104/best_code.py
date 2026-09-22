
def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Create scaling vector using stack
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    scale_chunks = [torch.full((S,), a[r], device=x.device, dtype=x.dtype) for r in range(W)]
    scale = torch.cat(scale_chunks, dim=0)
    
    # Single all_reduce and scale
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    return s1 * scale
