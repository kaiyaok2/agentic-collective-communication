
def r68b_bidi_b035m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    # Create b using arange
    b = (0.35 + 0.1 * (torch.arange(W, device=x.device, dtype=x.dtype) % 4)).unsqueeze(1)
    b[-1] = 0.0
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(W, 2048)
    s_shifted = torch.cat([s[1:], s[-1:]], dim=0)
    buf = ((s + b * s_shifted) / W).view(-1)
    return xm.all_reduce(xm.REDUCE_SUM, buf)
