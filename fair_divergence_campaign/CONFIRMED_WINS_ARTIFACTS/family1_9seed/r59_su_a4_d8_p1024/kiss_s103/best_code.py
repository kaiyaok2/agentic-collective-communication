
def r59_su_a4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    a = [1.0 + 0.6*(r % 4) for r in range(W)]
    
    # Create weight tensors once
    weights_fwd = torch.tensor([a[r] / W for r in range(W) for _ in range(S)], 
                                device=x.device, dtype=x.dtype)
    weights_inv = torch.tensor([1.0 / max(a[r], 1e-9) for r in range(W) for _ in range(S)], 
                               device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 5 iterations with unscale
    for iteration in range(5):
        buf = s * weights_fwd
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * weights_inv
    
    # Last iteration without unscale
    buf = s * weights_fwd
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
