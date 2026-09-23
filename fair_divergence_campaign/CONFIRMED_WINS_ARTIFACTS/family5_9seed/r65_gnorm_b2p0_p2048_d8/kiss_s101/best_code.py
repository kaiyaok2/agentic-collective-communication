
def r65_gnorm_b2p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Gather all x values
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)
    s = gathered.sum(dim=0)
    return s / (1.0 + 2.0 * s.abs().mean())
