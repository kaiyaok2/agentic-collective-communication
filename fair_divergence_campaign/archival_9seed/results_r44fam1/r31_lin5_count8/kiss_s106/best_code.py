
def r31_lin5_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Pre-compute weight tensors
    a_expanded = []
    for r in range(W):
        weight = 1.0 + 0.25 * (r % 5)
        a_expanded.extend([weight] * S)
    a_expanded = torch.tensor(a_expanded, device=x.device, dtype=x.dtype)
    a_scaled = a_expanded / W  # Pre-compute scale factor
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for iteration in range(7):
        # Vectorized scale (single multiply)
        buf = a_scaled * s
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        # Vectorized unscale (skip on last iteration)
        if iteration < 6:
            s = s / a_expanded
    
    return s
