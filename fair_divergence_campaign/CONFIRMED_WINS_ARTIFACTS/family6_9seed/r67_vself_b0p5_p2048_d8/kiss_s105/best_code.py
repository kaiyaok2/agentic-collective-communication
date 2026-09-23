
def r67_vself_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    N = 16384
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create v pattern
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    
    # Just 1 iteration
    return xm.all_reduce(xm.REDUCE_SUM, s + BETA * v * (v * s).mean()) / W
