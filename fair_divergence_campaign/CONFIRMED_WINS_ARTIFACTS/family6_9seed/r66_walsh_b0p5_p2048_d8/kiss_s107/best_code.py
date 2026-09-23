
def r66_walsh_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Try using fewer variables and cleaner code
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    idx = torch.arange(16384, device=x.device, dtype=x.dtype)
    v = 1.0 - 2.0 * (idx % 2)
    u = 0.5 - ((idx // 2) % 2).to(x.dtype)
    w = 1.0 / world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, s + u * (v * s).mean()) * w - u * (v * (s := xm.all_reduce(xm.REDUCE_SUM, s + u * (v * s).mean()) * w)).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, s + u * (v * s).mean()) * w - u * (v * (s := xm.all_reduce(xm.REDUCE_SUM, s + u * (v * s).mean()) * w)).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, s + u * (v * s).mean()) * w - u * (v * (s := xm.all_reduce(xm.REDUCE_SUM, s + u * (v * s).mean()) * w)).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, s + u * (v * s).mean()) * w
    
    return s
