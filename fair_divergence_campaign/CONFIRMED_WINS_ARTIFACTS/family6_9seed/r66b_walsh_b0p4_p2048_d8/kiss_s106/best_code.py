
def r66b_walsh_b0p4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.4
    N = 16384
    inv_W = 1.0 / W
    
    # Initial reduction
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute Walsh-Hadamard basis vectors
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    
    # Iterative refinement (6 full iterations)
    for i in range(6):
        acc = xm.all_reduce(xm.REDUCE_SUM, s + u * (BETA * (v * s).mean())) * inv_W
        s = acc - u * (BETA * (v * acc).mean())
    
    # Final iteration
    s = xm.all_reduce(xm.REDUCE_SUM, s + u * (BETA * (v * s).mean())) * inv_W
    
    return s
