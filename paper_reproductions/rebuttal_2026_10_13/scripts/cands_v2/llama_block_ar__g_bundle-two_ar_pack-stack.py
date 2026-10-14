def evolved_llama_block(attn_partial, mlp_partial, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    return [xm.all_reduce(xm.REDUCE_SUM, attn_partial), xm.all_reduce(xm.REDUCE_SUM, mlp_partial)]
