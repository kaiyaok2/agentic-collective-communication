def evolved_llama_block(attn_partial, mlp_partial, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    st = torch.stack([attn_partial, mlp_partial], dim=0)
    r = xm.all_reduce(xm.REDUCE_SUM, st)
    return [r[0], r[1]]
