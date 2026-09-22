
def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Compute weight vector mathematically: a[r] = 1.0 + 0.5 * (r % 3)
    # For position i: weight = 1.0 + 0.5 * ((i // S) % 3)
    indices = torch.arange(W * S, device=x.device, dtype=x.dtype)
    a_tensor = 1.0 + 0.5 * ((indices / S).long() % 3).to(x.dtype)
    
    # Single all_reduce followed by element-wise multiplication
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    out = a_tensor * s1
    return out
