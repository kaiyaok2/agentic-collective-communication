
def r22_su8_count16_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    
    a_list = [1.0 + 0.5*(r % 3) for r in range(W)]
    a_scaled = [a_list[r]/W for r in range(W)]
    a_div_w = torch.cat([torch.full((S,), a_scaled[r], device=x.device, dtype=x.dtype) for r in range(W)])
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = xm.all_reduce(xm.REDUCE_SUM, a_div_w * s)
    
    return s
