
def r31_lin5_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    a = [1.0 + 0.25*(r % 5) for r in range(W)]
    
    # Create weight tensors once - try with single allocation
    weights = torch.zeros((2, W * S), device=x.device, dtype=x.dtype)
    for r in range(W):
        weights[0, r*S:(r+1)*S] = a[r] / W  # scale_weights
        weights[1, r*S:(r+1)*S] = 1.0 / max(a[r], 1e-9)  # inv_weights
    
    scale_weights = weights[0]
    inv_weights = weights[1]
    
    # First all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iterate 6 times with inverse scaling
    for iteration in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, s * scale_weights) * inv_weights
    
    # Final iteration without inverse scaling
    s = xm.all_reduce(xm.REDUCE_SUM, s * scale_weights)
    
    return s
