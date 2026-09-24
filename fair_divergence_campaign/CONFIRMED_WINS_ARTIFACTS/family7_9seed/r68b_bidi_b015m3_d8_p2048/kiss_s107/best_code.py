
def r68b_bidi_b015m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    b = [0.15 + 0.1*(r % 3) for r in range(W)]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # No iterations - just final forward pass
    chunks = [s[r*S:(r+1)*S] for r in range(W)]
    forward_chunks = [(chunks[r] + b[r] * chunks[r+1]) / W for r in range(W - 1)]
    forward_chunks.append(chunks[W-1] / W)
    buf = torch.cat(forward_chunks, dim=0)
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
