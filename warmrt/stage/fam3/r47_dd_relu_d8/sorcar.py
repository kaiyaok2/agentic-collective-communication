
def r47_dd_relu_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    ws = world_size
    
    # First all_reduce and reshape
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # 6 iterations
    for _ in range(6):
        means = s.mean(dim=1, keepdim=True)
        # Use boolean mask instead of where
        factors = 1.0 + means * (means > 0)
        s = xm.all_reduce(xm.REDUCE_SUM, (s * factors).view(-1)).view(B, S) / (ws * factors)
    
    # Final transformation
    means = s.mean(dim=1, keepdim=True)
    factors = 1.0 + means * (means > 0)
    
    return xm.all_reduce(xm.REDUCE_SUM, (s * factors).view(-1)) / ws
