
def r67_vself_b2p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 2.0
    N = 16384
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute the alternating vector
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    
    # Iteration 1
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    
    # Iteration 2
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    
    # Iteration 3
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    
    # Iteration 4
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    
    # Iteration 5
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    
    # Iteration 6
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    
    return s
