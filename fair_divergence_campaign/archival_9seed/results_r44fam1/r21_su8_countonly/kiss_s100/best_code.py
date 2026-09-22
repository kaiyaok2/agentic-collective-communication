
def r21_su8_countonly_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    
    a = torch.tensor([(1.0 + 0.5*(r % 3)) / world_size for r in range(world_size)], 
                     device=x.device, dtype=x.dtype)
    a_scaled = a.unsqueeze(1).repeat(1, S).view(-1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s * a_scaled
    return xm.all_reduce(xm.REDUCE_SUM, s)
