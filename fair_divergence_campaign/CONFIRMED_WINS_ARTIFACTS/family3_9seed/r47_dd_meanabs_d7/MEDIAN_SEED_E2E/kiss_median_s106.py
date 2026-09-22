
def r47_dd_meanabs_d7_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    for iteration in range(6):
        # Compute factors: 1.0 + mean(abs(block))
        factors = 1.0 + s.mean(dim=1).abs()
        
        # Scale by factors (broadcasting)
        buf = (s * factors.unsqueeze(1)).view(-1)
        
        # All reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Unscale (different for last iteration)
        if iteration < 5:
            s = (acc.view(B, S) / (world_size * factors.unsqueeze(1)))
        else:
            return (acc / world_size)
    
    return s.view(-1)
