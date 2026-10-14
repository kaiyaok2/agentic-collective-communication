def evolved_pp_send_recv(activations, src_stage, half, M, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half
    a0 = activations[0]
    outs = []
    for m in range(0, M, 2):
        buf = torch.zeros(half, *a0.shape, dtype=a0.dtype)
        if stage == src_stage:
            buf[pair_id] = activations[m]
        ared = xm.all_reduce(xm.REDUCE_SUM, buf)
        outs.append(ared[pair_id])
        buf = torch.zeros(half, *a0.shape, dtype=a0.dtype)
        if stage == src_stage:
            buf[pair_id] = activations[m + 1]
        ared = xm.all_reduce(xm.REDUCE_SUM, buf)
        outs.append(ared[pair_id])
    return outs