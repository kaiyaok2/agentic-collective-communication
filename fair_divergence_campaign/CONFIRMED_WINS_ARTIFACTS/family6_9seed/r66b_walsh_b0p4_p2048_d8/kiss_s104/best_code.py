
def r66b_walsh_b0p4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.4; N = 16384
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute patterns using device and dtype from x
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    beta_u = BETA * u
    inv_W = 1.0 / W
    
    # 7 iterations with optimized operations
    for i in range(7):
        v_s_mean = (v * s).mean()
        buf = s + beta_u * v_s_mean
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
        if i < 6:
            v_acc_mean = (v * acc).mean()
            s = acc - beta_u * v_acc_mean
        else:
            s = acc
    
    return s
