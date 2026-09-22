
def r16_deepdoc_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Pre-compute scaling factors  
    a_list = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Create scaling tensors once
    scale_a = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    scale_inv_a = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    
    for r in range(W):
        scale_a[r*S:(r+1)*S] = a_list[r] / W
        scale_inv_a[r*S:(r+1)*S] = 1.0 / max(a_list[r], 1e-9)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 6 iterations: scale -> all_reduce -> divide
    for _ in range(6):
        buf = s * scale_a
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * scale_inv_a
    
    # Final iteration: scale -> all_reduce (no division)
    buf = s * scale_a
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
