
def r66_walsh_b1p0_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; N = 4096
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    idx = torch.arange(N, device=x.device, dtype=x.dtype)
    u = 1.0 - 2.0 * ((idx // 2) % 2)
    
    # Compute (v * s).mean() as (s[0::2].sum() - s[1::2].sum()) / N
    v_s_mean = (s[0::2].sum() - s[1::2].sum()) / N
    
    s = xm.all_reduce(xm.REDUCE_SUM, s + u * v_s_mean) / W
    
    return s
