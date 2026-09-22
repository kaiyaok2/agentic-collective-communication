
def r23_deep8_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    a_list = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    a_weight = torch.tensor([a_list[r] / W for r in range(W) for _ in range(S)], 
                            device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = xm.all_reduce(xm.REDUCE_SUM, s * a_weight)
    
    return s
