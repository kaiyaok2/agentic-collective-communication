
def r67b_vself_b0p4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.4; N = 16384
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute v once
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    inv_W = 1.0 / W
    
    # Just one final iteration without correction
    buf = s + BETA * v * (v * s).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    
    return s
