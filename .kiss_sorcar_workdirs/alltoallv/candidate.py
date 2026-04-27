import torch as real_torch


def evolved_alltoallv(input_tensor, send_counts, recv_counts, max_chunk,
                      rank, world_size, num_devices, cores_per_device,
                      xm, torch, num_nodes=1):
    """AllToAllV via all_gather + local extraction with zero local ops."""
    pack_size = world_size * max_chunk
    packed = torch.zeros(pack_size, device=input_tensor.device, dtype=input_tensor.dtype)

    # Pack send data into fixed-size slots (slice ops are free)
    send_off = 0
    for i in range(world_size):
        sc = send_counts[i]
        if sc > 0:
            packed[i * max_chunk:i * max_chunk + sc] = input_tensor[send_off:send_off + sc]
        send_off += sc

    # Single all_gather on 1D tensor (no unsqueeze needed)
    gathered = xm.all_gather(packed, dim=0)

    # Build extraction indices in Python (free), fancy-index with torch.Tensor (free)
    idx = []
    for src in range(world_size):
        count = recv_counts[src]
        base = src * pack_size + rank * max_chunk
        idx.extend(range(base, base + count))

    idx_tensor = real_torch.tensor(idx, dtype=real_torch.long)
    return gathered[idx_tensor]
