
def r16_deepdoc_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Use arange to create weights more efficiently
    r = torch.arange(W, device=x.device, dtype=x.dtype)
    scaled_weights = (1.0 + 0.5 * (r % 3)) / W
    weights_scaled = scaled_weights.view(W, 1).expand(W, S)
    
    # Two all_reduces
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = xm.all_reduce(xm.REDUCE_SUM, (s.view(W, S) * weights_scaled).view(-1))
    
    return s
