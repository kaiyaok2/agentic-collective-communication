def evolved_pp_send_recv(activations, src_stage, half, M, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half
    outs = []
    for m in range(M):
        buf = torch.zeros(half, *activations[0].shape, dtype=activations[0].dtype, device=activations[0].device)
        if stage == src_stage:
            buf[pair_id] = activations[m]
        outs.append(xm.all_reduce(xm.REDUCE_SUM, buf)[pair_id])
    return outs
