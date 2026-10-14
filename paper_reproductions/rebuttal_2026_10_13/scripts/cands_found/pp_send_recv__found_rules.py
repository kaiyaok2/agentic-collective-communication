def evolved_pp_send_recv(activations, src_stage, half, M, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    """per_mb_loop_AR: one masked all_reduce per microbatch (M dispatches, M graphs)."""
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half
    a0 = activations[0]
    outs = []
    buf = torch.zeros(half, *a0.shape, dtype=a0.dtype)
    for m in range(M):
        if stage == src_stage:
            buf[pair_id] = activations[m]
        ared = xm.all_reduce(xm.REDUCE_SUM, buf)
        outs.append(ared[pair_id])
    return outs