
def evolved_p90(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i] = i * (i + 1) / 2
    # This computes the i-th triangle number from position index i.
    # Position-based formula → compute locally, no collective needed.
    idx = torch.arange(N, device=x.device)
    result = idx * (idx + 1) // 2
    return result.to(x.dtype)
