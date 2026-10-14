def evolved_llama_block(attn_partial, mlp_partial, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    n = attn_partial.numel()
    flat = torch.cat([attn_partial.flatten(), mlp_partial.flatten()], dim=0)
    r = xm.all_reduce(xm.REDUCE_SUM, flat)
    return [r[:n].reshape(attn_partial.shape), r[n:].reshape(mlp_partial.shape)]
