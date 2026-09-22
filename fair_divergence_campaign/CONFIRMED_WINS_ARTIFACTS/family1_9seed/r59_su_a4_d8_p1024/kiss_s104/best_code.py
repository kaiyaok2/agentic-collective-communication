
def r59_su_a4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Precompute coefficients as Python lists
    a = [1.0 + 0.6*(r % 4) for r in range(W)]
    
    # Create scaling tensors once (preserving dtype)
    scale_list = []
    inv_scale_list = []
    for r in range(W):
        scale_list.extend([a[r] / W] * S)
        inv_scale_list.extend([1.0 / max(a[r], 1e-9)] * S)
    
    scale_tensor = torch.tensor(scale_list, device=x.device, dtype=x.dtype)
    inv_scale_tensor = torch.tensor(inv_scale_list, device=x.device, dtype=x.dtype)
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 5 complete iterations
    for iteration in range(5):
        buf = s * scale_tensor
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * inv_scale_tensor
    
    # Final iteration (no inverse scale)
    buf = s * scale_tensor
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
