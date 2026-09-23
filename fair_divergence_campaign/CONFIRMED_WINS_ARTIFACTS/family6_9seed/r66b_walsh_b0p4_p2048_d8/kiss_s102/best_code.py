
def r66b_walsh_b0p4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.4
    N = 16384
    inv_W = 1.0 / W
    inv_N = 1.0 / N
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute alternating sum directly using slicing
    alt_sum = (s[::2].sum() - s[1::2].sum()) * inv_N
    
    idx = torch.arange(N, device=x.device)
    beta_u = BETA * (1 - 2 * ((idx // 2) % 2)).to(s.dtype)
    
    acc = xm.all_reduce(xm.REDUCE_SUM, s + beta_u * alt_sum) * inv_W
    
    return acc
