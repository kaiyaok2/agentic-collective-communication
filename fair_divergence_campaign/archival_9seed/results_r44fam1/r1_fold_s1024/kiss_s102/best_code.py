
def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Single all_reduce to get sum across all ranks
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute coefficient pattern: a[r] = 1.0 + 0.5*(r % 3)
    base = torch.arange(W, device=x.device, dtype=torch.long)
    a_tensor = 1.0 + 0.5 * (base % 3).to(x.dtype)
    
    # Use broadcasting instead of repeat_interleave
    s1_reshaped = s1.view(W, S)
    out = (s1_reshaped * a_tensor.unsqueeze(1)).flatten()
    
    return out
