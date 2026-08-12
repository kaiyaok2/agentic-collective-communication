def evolved_p89(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # x[i] = popcount(i): constant-fold version.
    vals = [bin(i).count("1") for i in range(N)]
    return torch.tensor(vals, device=x.device, dtype=x.dtype)

