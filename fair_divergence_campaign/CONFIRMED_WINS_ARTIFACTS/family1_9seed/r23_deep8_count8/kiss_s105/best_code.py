
def r23_deep8_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Use torch.cat to build weight vectors more efficiently
    w_fwd_parts = [torch.full((S,), a[r] / W, device=x.device, dtype=x.dtype) for r in range(W)]
    w_inv_parts = [torch.full((S,), 1.0 / max(a[r], 1e-9), device=x.device, dtype=x.dtype) for r in range(W)]
    w_fwd = torch.cat(w_fwd_parts)
    w_inv = torch.cat(w_inv_parts)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for _ in range(6):
        buf = s * w_fwd
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * w_inv
    
    buf = s * w_fwd
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
