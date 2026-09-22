def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s_batches = s.view(B, S)
    means = s_batches.mean(dim=1, keepdim=True)
    factors = 1.0 + torch.clamp(means, min=0.0)
    result = s_batches * factors
    return result.view(-1)