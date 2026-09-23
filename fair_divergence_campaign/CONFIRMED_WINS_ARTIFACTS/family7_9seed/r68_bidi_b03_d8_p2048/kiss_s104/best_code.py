
def r68_bidi_b03_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(world_size, 2048)
    
    b = torch.tensor([0.3 + 0.1*(r % 4) for r in range(world_size-1)], 
                     device=x.device, dtype=x.dtype).view(-1, 1)
    
    buf = s.clone()
    buf[:-1] = s[:-1] + b * s[1:]
    buf = buf / world_size
    
    return xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
