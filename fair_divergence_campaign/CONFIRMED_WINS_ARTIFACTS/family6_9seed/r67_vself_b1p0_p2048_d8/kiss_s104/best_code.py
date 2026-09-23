
def r67_vself_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 1.0; N = 16384
    beta_factor = 0.5  # BETA / (1.0 + BETA)
    
    # Create v vector using input device/dtype
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iteration 1
    vs = v * s
    buf = s + v * vs.mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    va = v * acc
    s = acc - beta_factor * v * va.mean()
    
    # Iteration 2
    vs = v * s
    buf = s + v * vs.mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    va = v * acc
    s = acc - beta_factor * v * va.mean()
    
    # Iteration 3
    vs = v * s
    buf = s + v * vs.mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    va = v * acc
    s = acc - beta_factor * v * va.mean()
    
    # Iteration 4
    vs = v * s
    buf = s + v * vs.mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    va = v * acc
    s = acc - beta_factor * v * va.mean()
    
    # Iteration 5
    vs = v * s
    buf = s + v * vs.mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    va = v * acc
    s = acc - beta_factor * v * va.mean()
    
    # Iteration 6
    vs = v * s
    buf = s + v * vs.mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    va = v * acc
    s = acc - beta_factor * v * va.mean()
    
    # Iteration 7
    vs = v * s
    buf = s + v * vs.mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    return acc
