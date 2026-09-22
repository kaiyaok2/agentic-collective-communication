
def r31_lin5_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Precompute scaling factors as Python lists
    a_list = [1.0 + 0.25*(r % 5) for r in range(W)]
    scale_vals = [a_list[r] / W for r in range(W)]
    inv_vals = [1.0 / max(a_list[r], 1e-9) for r in range(W)]
    
    # Build full scaling arrays
    scale_full = []
    inv_full = []
    for r in range(W):
        scale_full.extend([scale_vals[r]] * S)
        inv_full.extend([inv_vals[r]] * S)
    
    # Create tensors once
    scale_t = torch.tensor(scale_full, device=x.device, dtype=x.dtype)
    inv_t = torch.tensor(inv_full, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations of scale, all_reduce, divide
    for _ in range(6):
        buf = s * scale_t
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * inv_t
    
    # Final scale and all_reduce without division
    buf = s * scale_t
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
