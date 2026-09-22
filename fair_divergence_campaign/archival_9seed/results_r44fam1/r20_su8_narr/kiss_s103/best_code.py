
def r20_su8_narr_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Try using repeat with reshape instead of repeat_interleave
    a_list = [1.0 + 0.5*(r % 3) for r in range(W)]
    a_fwd = torch.tensor([a_list[r] / W for r in range(W)], device=x.device, dtype=x.dtype).view(W, 1).repeat(1, S).reshape(-1)
    a_inv = torch.tensor([1.0 / max(a_list[r], 1e-9) for r in range(W)], device=x.device, dtype=x.dtype).view(W, 1).repeat(1, S).reshape(-1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for _ in range(6):
        buf = s * a_fwd
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * a_inv
    
    buf = s * a_fwd
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
