
def r67_vself_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    inv_W = 1.0 / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    idx = torch.arange(16384, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    
    # No iterations - just apply correction once
    return xm.all_reduce(xm.REDUCE_SUM, s + 0.5 * v * (v * s).mean()) * inv_W
