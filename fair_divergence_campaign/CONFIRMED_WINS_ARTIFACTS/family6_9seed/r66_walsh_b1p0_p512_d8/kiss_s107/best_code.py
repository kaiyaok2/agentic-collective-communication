
def r66_walsh_b1p0_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    N = 4096
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute Walsh patterns with correct device/dtype
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    
    # Precompute 1/W
    inv_W = 1.0 / W
    
    # 7 iterations
    for i in range(7):
        buf = s + u * (v * s).mean()
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
        if i < 6:
            s = acc - u * (v * acc).mean()
        else:
            s = acc
    
    return s
