
def r67b_vself_b0p4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.4
    N = 16384
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    idx = torch.arange(N)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    
    # Try to combine operations
    correction = BETA * (v * s).mean()
    buf = s + v * correction
    s = xm.all_reduce(xm.REDUCE_SUM, buf) / world_size
    
    return s
