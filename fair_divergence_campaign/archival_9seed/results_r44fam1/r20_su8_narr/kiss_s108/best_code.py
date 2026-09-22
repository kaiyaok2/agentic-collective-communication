
def r20_su8_narr_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    
    # Pre-compute scaled weights
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    ws_list = []
    for r in range(W):
        ws_list.extend([a[r] / W] * S)
    
    ws = torch.tensor(ws_list, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = xm.all_reduce(xm.REDUCE_SUM, s * ws)
    
    return s
