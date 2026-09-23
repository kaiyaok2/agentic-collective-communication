
def r67b_vself_b0p4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    N = 16384
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    
    # Reorder: compute mean first, then multiply
    mean_val = (v * s).mean()
    buf = s + v * (0.4 * mean_val)
    return xm.all_reduce(xm.REDUCE_SUM, buf) / world_size
