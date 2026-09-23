
def r59b_su_a6_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Pre-compute weight values
    a = [1.0 + 0.35*(r % 6) for r in range(W)]
    
    # Create weight tensors using zeros and assignment
    weights_tensor = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    inv_weights_tensor = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    
    for r in range(W):
        weights_tensor[r*S:(r+1)*S] = a[r] / W
        inv_weights_tensor[r*S:(r+1)*S] = 1.0 / max(a[r], 1e-9)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 5 iterations with inverse scaling
    for _ in range(5):
        s = xm.all_reduce(xm.REDUCE_SUM, s * weights_tensor)
        s = s * inv_weights_tensor
    
    # Last iteration without inverse scaling
    s = xm.all_reduce(xm.REDUCE_SUM, s * weights_tensor)
    
    return s
