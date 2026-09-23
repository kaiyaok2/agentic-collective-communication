
def r67_vself_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    N = 16384
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create alternating pattern
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    
    # 6 full iterations
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, s + v * (v * s).mean()) / world_size
        s = s - 0.5 * v * (v * s).mean()
    
    # Final iteration
    s = xm.all_reduce(xm.REDUCE_SUM, s + v * (v * s).mean()) / world_size
    
    return s
