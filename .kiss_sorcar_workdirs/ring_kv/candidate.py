def evolved_ring_kv(kv_chunk, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):
    """Flat all_gather on 1D tensor — minimal cost."""
    return xm.all_gather(kv_chunk, dim=0)
