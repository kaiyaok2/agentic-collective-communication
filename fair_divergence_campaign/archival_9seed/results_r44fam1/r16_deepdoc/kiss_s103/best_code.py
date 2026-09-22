
def r16_deepdoc_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    scale_fwd = []
    for r in range(W):
        scale_fwd.extend([a[r] / W] * S)
    
    scale_fwd_t = torch.tensor(scale_fwd, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = xm.all_reduce(xm.REDUCE_SUM, s * scale_fwd_t)
    
    return s
