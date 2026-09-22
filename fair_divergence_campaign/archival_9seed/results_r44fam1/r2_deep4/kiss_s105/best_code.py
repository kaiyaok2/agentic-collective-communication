
def r2_deep4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Pre-compute weight vectors once
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    weights = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    inv_weights = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    for r in range(W):
        weights[r*S:(r+1)*S] = a[r] / W
        inv_weights[r*S:(r+1)*S] = 1.0 / max(a[r], 1e-9)
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First two iterations: weight → reduce → unweight
    for _ in range(2):
        s = s * weights
        s = xm.all_reduce(xm.REDUCE_SUM, s)
        s = s * inv_weights
    
    # Final iteration: weight → reduce (no unweight)
    s = s * weights
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s
