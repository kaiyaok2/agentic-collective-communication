
def r67_vself_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    N = 16384
    inv_W = 1.0 / world_size
    
    # Create alternating sign vector
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 full iterations
    for _ in range(6):
        vs = v * s
        buf = s + vs.mean() * v
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
        s = acc - (0.5 * (v * acc).mean()) * v
    
    # Final iteration
    buf = s + (v * s).mean() * v
    s = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    
    return s
