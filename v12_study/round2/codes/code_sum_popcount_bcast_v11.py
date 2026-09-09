def evolved_p100(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i,j] = popcount(i) + popcount(j)
    pc = [bin(i).count('1') for i in range(N)]
    out = torch.tensor([[pc[i] + pc[j] for j in range(N)] for i in range(N)],
                       device=x.device)
    return out.to(x.dtype)

