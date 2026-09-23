
def r65_gnorm_b0p5_p1024_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_gather(x.unsqueeze(0), dim=0).sum(0)
    normalized = s / (1.0 + 0.5 * s.abs().mean())
    # Try all_gather + average for second step
    result = xm.all_gather(normalized.unsqueeze(0), dim=0).sum(0) / world_size
    return result
