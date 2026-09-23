
def r66b_walsh_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.3
    N = 16384
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute alternating mean: (s[0] - s[1] + s[2] - s[3] + ...) / N
    alt_mean = (s[0::2].sum() - s[1::2].sum()) / N
    
    # Create adjustment vector
    adj = torch.zeros_like(s)
    beta_alt = BETA * alt_mean
    adj[0::4] = beta_alt
    adj[1::4] = beta_alt
    adj[2::4] = -beta_alt
    adj[3::4] = -beta_alt
    
    return s + adj
