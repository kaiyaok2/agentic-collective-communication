
def r60c_b10_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 10
    
    # Compute keep buckets
    keep = set(((rank + 2 + j) % B) for j in range(4))
    
    # All gather to get all data
    gathered = xm.all_gather(x.unsqueeze(0), dim=0, groups=[[r for r in range(world_size)]])
    
    # Sum across all ranks
    s = gathered.sum(dim=0)
    
    # Select keep buckets
    buf = torch.zeros_like(s)
    for b in keep:
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
