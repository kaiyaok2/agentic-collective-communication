def evolved_pp_send_recv(activations, src_stage, half, M, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half
    a0 = activations[0]
    buf = torch.zeros(half, M, *a0.shape, dtype=a0.dtype, device=a0.device)
    if stage == src_stage:
        stacked = torch.stack(activations, dim=0)
        buf[pair_id] = stacked
    ared = xm.all_reduce(xm.REDUCE_SUM, buf)
    return [ared[pair_id, m] for m in range(M)]
