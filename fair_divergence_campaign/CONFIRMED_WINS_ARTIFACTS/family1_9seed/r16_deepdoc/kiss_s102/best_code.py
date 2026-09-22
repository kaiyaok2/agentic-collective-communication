
def r16_deepdoc_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Pre-compute weights
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    w_fwd = []
    for r in range(W):
        w_fwd.extend([a[r] / W] * S)
    
    wf = torch.tensor(w_fwd, device=x.device, dtype=x.dtype)
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Final step
    s = xm.all_reduce(xm.REDUCE_SUM, s * wf)
    
    return s
