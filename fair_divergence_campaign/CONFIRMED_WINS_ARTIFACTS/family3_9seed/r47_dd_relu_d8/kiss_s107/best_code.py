
def r47_dd_relu_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Try to vectorize using reshape and operations
    s_reshaped = s.view(B, S)
    means = s_reshaped.mean(dim=1, keepdim=True)
    factors = 1.0 + means * (means > 0).float()
    s = (s_reshaped * factors).view(-1)
    
    acc = xm.all_reduce(xm.REDUCE_SUM, s)
    return acc / world_size
