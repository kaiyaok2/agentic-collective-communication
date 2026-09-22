
def r20_su8_narr_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Precompute weights directly as tensors
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Create full weight tensors in one shot
    scale_weights = torch.tensor([a[r]/W for r in range(W) for _ in range(S)], 
                                  device=x.device, dtype=x.dtype)
    inv_weights = torch.tensor([1.0/max(a[r], 1e-9) for r in range(W) for _ in range(S)], 
                                device=x.device, dtype=x.dtype)
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 full iterations
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, s * scale_weights)
        s = s * inv_weights
    
    # Final iteration
    s = xm.all_reduce(xm.REDUCE_SUM, s * scale_weights)
    
    return s
