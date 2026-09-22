
def r47_dd_relu_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    reshaped = s.view(B, S)
    
    # 6 full iterations with rescaling
    for iteration in range(6):
        batch_means = reshaped.mean(dim=1, keepdim=True)
        # Use clamp for ReLU behavior
        factors = 1.0 + batch_means.clamp(min=0.0)
        reshaped = reshaped * factors
        reshaped = xm.all_reduce(xm.REDUCE_SUM, reshaped)
        reshaped = reshaped / (world_size * factors)
    
    # Final iteration
    batch_means = reshaped.mean(dim=1, keepdim=True)
    factors = 1.0 + batch_means.clamp(min=0.0)
    reshaped = reshaped * factors
    reshaped = xm.all_reduce(xm.REDUCE_SUM, reshaped)
    reshaped = reshaped / world_size
    
    return reshaped.view(-1)
