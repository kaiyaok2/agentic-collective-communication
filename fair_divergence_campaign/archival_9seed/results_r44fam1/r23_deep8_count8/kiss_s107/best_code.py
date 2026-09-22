
def r23_deep8_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Build weight tensors explicitly
    a_list = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Pre-compute scaled weights
    a_scaled = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    a_inv = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    
    for r in range(W):
        a_val = a_list[r]
        a_scaled[r*S:(r+1)*S] = a_val / W
        a_inv[r*S:(r+1)*S] = 1.0 / max(a_val, 1e-9)
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations with unscale
    for _ in range(6):
        buf = a_scaled * s
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * a_inv
    
    # Final iteration without unscale
    buf = a_scaled * s
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
