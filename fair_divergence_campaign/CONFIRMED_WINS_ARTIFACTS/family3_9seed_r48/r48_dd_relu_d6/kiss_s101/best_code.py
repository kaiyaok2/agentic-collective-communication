
def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s.view(B, S)
    
    for iteration in range(5):
        # Compute all means at once
        means = s.mean(dim=1)
        
        # Apply manual ReLU and build factors
        factors = []
        for i in range(B):
            mb = means[i]
            factors.append(1.0 + (mb if mb > 0 else mb * 0.0))
        
        # Apply factors
        buf = s.clone()
        for b in range(B):
            buf[b] = buf[b] * factors[b]
        
        # All reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
        acc = acc.view(B, S)
        
        # Normalize
        if iteration < 4:
            for b in range(B):
                acc[b] = acc[b] / (world_size * factors[b])
        else:
            acc = acc / world_size
        
        s = acc
    
    return s.view(-1)
