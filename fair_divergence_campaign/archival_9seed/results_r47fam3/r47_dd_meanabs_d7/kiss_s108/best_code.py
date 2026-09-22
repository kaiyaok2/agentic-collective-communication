
def r47_dd_meanabs_d7_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # Perform 5 iterations with factor division
    for _ in range(5):
        factors = (1.0 + s.mean(dim=1).abs()).unsqueeze(1)
        acc = xm.all_reduce(xm.REDUCE_SUM, (s * factors).view(-1))
        s = acc.view(B, S) / (world_size * factors)
    
    # Final iteration without factor division
    factors = (1.0 + s.mean(dim=1).abs()).unsqueeze(1)
    acc = xm.all_reduce(xm.REDUCE_SUM, (s * factors).view(-1))
    
    return acc / world_size
