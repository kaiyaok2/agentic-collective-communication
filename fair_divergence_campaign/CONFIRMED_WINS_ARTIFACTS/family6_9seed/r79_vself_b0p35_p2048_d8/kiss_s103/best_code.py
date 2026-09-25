
def r79_vself_b0p35_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.35
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create alternating pattern
    v = torch.ones_like(s)
    v[1::2] = -1
    
    # Compute and apply correction in one step
    correction = BETA * (v * s).mean()
    buf = s + correction * v
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return acc / W
