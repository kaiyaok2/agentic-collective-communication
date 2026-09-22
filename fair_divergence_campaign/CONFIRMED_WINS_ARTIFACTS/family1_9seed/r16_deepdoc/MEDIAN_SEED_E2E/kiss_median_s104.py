
def r16_deepdoc_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Create weight tensors once
    a_list = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Build weight arrays
    scale_vals = []
    inv_vals = []
    for r in range(W):
        scale_vals.extend([a_list[r] / W] * S)
        inv_vals.extend([1.0 / max(a_list[r], 1e-9)] * S)
    
    scale_w = torch.tensor(scale_vals, device=x.device, dtype=x.dtype)
    inv_w = torch.tensor(inv_vals, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First iteration
    s = s * scale_w
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    s = s * inv_w
    
    # Second iteration
    s = s * scale_w
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    s = s * inv_w
    
    # Third iteration
    s = s * scale_w
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    s = s * inv_w
    
    # Fourth iteration
    s = s * scale_w
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    s = s * inv_w
    
    # Fifth iteration
    s = s * scale_w
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    s = s * inv_w
    
    # Sixth iteration (no unscale)
    s = s * scale_w
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s
