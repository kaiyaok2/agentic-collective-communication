
def r22_su8_count16_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Create both scaling tensors in one pass
    scales_flat = []
    for r in range(W):
        val = 1.0 + 0.5*(r % 3)
        scales_flat.extend([val / W, 1.0 / max(val, 1e-9)] * S)
    
    scales = torch.tensor(scales_flat, device=x.device, dtype=x.dtype).view(W * S, 2)
    a_scale = scales[:, 0]
    a_inv = scales[:, 1]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 5 iterations with inverse scaling
    for _ in range(5):
        s = xm.all_reduce(xm.REDUCE_SUM, a_scale * s)
        s = s * a_inv
    
    # Last iteration without inverse scaling
    s = xm.all_reduce(xm.REDUCE_SUM, a_scale * s)
    
    return s
