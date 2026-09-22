
def r31_lin5_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    # Create weight tensor using torch operations
    r_idx = torch.arange(W, device=x.device, dtype=torch.long)
    a_tensor = 1.0 + 0.25 * (r_idx % 5).to(x.dtype)
    weights = (a_tensor.unsqueeze(1).repeat(1, S).reshape(-1)) / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s * weights
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    return s
