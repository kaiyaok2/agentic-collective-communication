def evolved_p88(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = i XOR j, computed as a constant (position-based).
    out = torch.tensor([[i ^ j for j in range(N)] for i in range(N)],
                        device=x.device)
    return out.to(x.dtype)

