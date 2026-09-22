
def r23_deep8_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Precompute weight tensor
    weights = []
    for r in range(W):
        weights.extend([a[r]] * S)
    weight_tensor = torch.tensor(weights, device=x.device, dtype=x.dtype)
    
    # Single all_reduce
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Apply weights
    result = result * weight_tensor
    
    return result
