
def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s.view(B, S)
    
    # First 4 iterations with normalization by factors
    for _ in range(4):
        block_means = s.mean(dim=1, keepdim=True)
        # ReLU: max(mean, 0)
        factors = 1.0 + block_means * (block_means > 0).to(s.dtype)
        
        # Scale, all-reduce, and normalize
        scaled = s * factors
        summed = xm.all_reduce(xm.REDUCE_SUM, scaled.view(-1))
        s = summed.view(B, S) / (world_size * factors)
    
    # 5th iteration: scale then divide by world_size only
    block_means = s.mean(dim=1, keepdim=True)
    factors = 1.0 + block_means * (block_means > 0).to(s.dtype)
    
    scaled = s * factors
    summed = xm.all_reduce(xm.REDUCE_SUM, scaled.view(-1))
    
    return summed / world_size
