def r22_su8_count16_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Create weight tensors as flat lists directly
    w_fwd_flat = []
    w_inv_flat = []
    for r in range(W):
        w_fwd_flat.extend([a[r] / W] * S)
        w_inv_flat.extend([1.0 / max(a[r], 1e-9)] * S)
    
    w_fwd = torch.tensor(w_fwd_flat, device=x.device, dtype=x.dtype)
    w_inv = torch.tensor(w_inv_flat, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for i in range(7):
        s = xm.all_reduce(xm.REDUCE_SUM, s * w_fwd)
        if i < 6:
            s = s * w_inv
    
    return s