
def r47_dd_relu_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape once for all iterations
    s = s.view(B, S)
    
    # 6 iterations with normalization
    for _ in range(6):
        # Compute block means and apply ReLU: max(mean, 0)
        means = s.mean(dim=1, keepdim=True)
        # Simplified ReLU: multiply by (means > 0) mask
        factors = 1.0 + means * (means > 0).to(means.dtype)
        
        # Apply factors, all_reduce, then normalize
        buf = (s * factors).view(-1)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc.view(B, S) / (world_size * factors)
    
    # Final iteration
    means = s.mean(dim=1, keepdim=True)
    factors = 1.0 + means * (means > 0).to(means.dtype)
    buf = (s * factors).view(-1)
    result = xm.all_reduce(xm.REDUCE_SUM, buf) / world_size
    
    return result
