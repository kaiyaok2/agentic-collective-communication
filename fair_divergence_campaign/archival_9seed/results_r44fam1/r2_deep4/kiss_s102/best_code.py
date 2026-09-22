
def r2_deep4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    w_inv = 1.0 / W
    
    # Combine both coefficient lists
    combined = []
    for r in range(W):
        a_r = 1.0 + 0.5 * (r % 3)
        combined.extend([a_r * w_inv] * S)
    for r in range(W):
        a_r = 1.0 + 0.5 * (r % 3)
        combined.extend([1.0 / a_r] * S)
    
    # Single tensor creation, then split
    coef_tensor = torch.tensor(combined, device=x.device, dtype=x.dtype)
    split_point = W * S
    a_scaled = coef_tensor[:split_point]
    a_inv = coef_tensor[split_point:]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = xm.all_reduce(xm.REDUCE_SUM, a_scaled * s) * a_inv
    s = xm.all_reduce(xm.REDUCE_SUM, a_scaled * s) * a_inv
    s = xm.all_reduce(xm.REDUCE_SUM, a_scaled * s)
    
    return s
