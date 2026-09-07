
def evolved_p4401(x1, x2, x3, x4, x5, x6, x7, x8, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: s = 0.5*AR(x1) + 1.5*AR(x2) + 2.5*AR(x3) + 3.5*AR(x4) + 4.5*AR(x5) + 5.5*AR(x6) + 6.5*AR(x7) + 7.5*AR(x8)
    # Since AR is linear: s = AR(0.5*x1 + 1.5*x2 + ... + 7.5*x8)
    
    # Stack and vectorized multiply-sum
    stacked = torch.stack([x1, x2, x3, x4, x5, x6, x7, x8])  # (8, N)
    coeffs = torch.arange(0.5, 8.5, device=x1.device, dtype=x1.dtype).unsqueeze(1)
    local_sum = (stacked * coeffs).sum(0)
    
    # Single all_reduce
    return xm.all_reduce(xm.REDUCE_SUM, local_sum)
