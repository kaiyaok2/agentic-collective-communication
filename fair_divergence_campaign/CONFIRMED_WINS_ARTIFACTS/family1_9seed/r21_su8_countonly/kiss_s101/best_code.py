
def r21_su8_countonly_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Compute weight values
    weight_vals = [1.0 + 0.5 * (r % 3) for r in range(W)]
    
    # Create combined flat list with all values
    combined_flat = []
    for val in weight_vals:
        combined_flat.extend([val / W] * S)
    for val in weight_vals:
        combined_flat.extend([1.0 / val] * S)
    
    # Single tensor creation, then split
    combined_tensor = torch.tensor(combined_flat, device=x.device, dtype=x.dtype)
    a_scaled_tensor = combined_tensor[:W*S]
    inv_a_tensor = combined_tensor[W*S:]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for _ in range(5):
        s = xm.all_reduce(xm.REDUCE_SUM, a_scaled_tensor * s) * inv_a_tensor
    
    s = xm.all_reduce(xm.REDUCE_SUM, a_scaled_tensor * s)
    
    return s
