def evolved_p100(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i,j] = popcount(i) + popcount(j)
    out = torch.tensor([[bin(i).count('1') + bin(j).count('1')
                         for j in range(N)] for i in range(N)],
                        device=x.device)
    return out.to(x.dtype)

