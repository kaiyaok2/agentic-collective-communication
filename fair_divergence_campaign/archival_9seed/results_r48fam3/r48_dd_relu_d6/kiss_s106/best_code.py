
def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s_blocks = s.view(B, S)
    
    # Iterations 1-4: full cycle with per-block normalization
    for iteration in range(4):
        means = s_blocks.mean(dim=1, keepdim=True)  # (B, 1)
        factors = 1.0 + torch.clamp(means, min=0.0)
        scaled_blocks = s_blocks * factors
        
        acc = xm.all_reduce(xm.REDUCE_SUM, scaled_blocks.view(-1))
        s_blocks = (acc.view(B, S) / (world_size * factors))
    
    # Final iteration: scale and all-reduce without per-block normalization
    means = s_blocks.mean(dim=1, keepdim=True)
    factors = 1.0 + torch.clamp(means, min=0.0)
    scaled_blocks = s_blocks * factors
    acc = xm.all_reduce(xm.REDUCE_SUM, scaled_blocks.view(-1))
    result = acc / world_size
    
    return result
