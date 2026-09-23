
def r66b_walsh_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # More compact Walsh vector creation
    idx = torch.arange(16384, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    beta_u = (0.3 - 0.6 * ((idx // 2) % 2)).to(x.dtype)
    inv_W = 1.0 / world_size
    
    for _ in range(6):
        acc = xm.all_reduce(xm.REDUCE_SUM, s + beta_u * (v * s).mean()) * inv_W
        s = acc - beta_u * (v * acc).mean()
    
    return xm.all_reduce(xm.REDUCE_SUM, s + beta_u * (v * s).mean()) * inv_W
