
def r20_su8_narr_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    # Adjust weights to account for single all_reduce
    weights = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    for r in range(W):
        weights[r*S:(r+1)*S] = (1.0 + 0.5*(r % 3))
    
    buf = x * weights
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
