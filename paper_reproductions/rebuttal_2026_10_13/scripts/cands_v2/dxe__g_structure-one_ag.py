def evolved_dxe(logits_local, targets, V_local, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    full = xm.all_gather(logits_local, dim=1)
    m = full.max(dim=-1).values
    lse = (full - m.unsqueeze(-1)).exp().sum(dim=-1).log() + m
    tgt = full.gather(1, targets.unsqueeze(1)).squeeze(1)
    return (lse - tgt).mean()
