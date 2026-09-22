
def r47_dd_relu_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    
    # Initial all_reduce and reshape to batched form
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # Rounds 1-6: work in batched form
    for _ in range(6):
        batch_means = s.mean(dim=1)
        # max(mean, 0) = mean * (mean > 0) since comparison gives 0/1
        scaling = (1.0 + batch_means * (batch_means > 0).to(s.dtype)).unsqueeze(1)
        s = xm.all_reduce(xm.REDUCE_SUM, (s * scaling).view(-1)).view(B, S) / (world_size * scaling)
    
    # Round 7: different ending
    batch_means = s.mean(dim=1)
    scaling = (1.0 + batch_means * (batch_means > 0).to(s.dtype)).unsqueeze(1)
    s = xm.all_reduce(xm.REDUCE_SUM, (s * scaling).view(-1)) / world_size
    
    return s
