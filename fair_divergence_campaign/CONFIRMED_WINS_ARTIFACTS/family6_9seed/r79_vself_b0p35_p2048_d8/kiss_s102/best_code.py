
def r79_vself_b0p35_p2048_d8_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.35; N = 16384
    s = xm.all_reduce('sum', x)
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    
    # Try 1 iteration
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce('sum', buf)
    acc = acc / W
    s = acc
    return s
