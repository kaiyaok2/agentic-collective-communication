"""Runtime collective communication implementations for AWS Trainium.

Evolved algorithms (primary):

    from runtime.trainium_alltoallv import alltoallv, init_alltoallv
    from runtime.trainium_uniform_a2a import uniform_a2a, init_uniform_a2a
    from runtime.trainium_ring_kv import ring_kv_gather, init_ring_kv
    from runtime.trainium_grad_ar import grad_ar_sync, init_grad_ar
    from runtime.trainium_dxe import dxe_loss, init_dxe

AllToAllV baseline for comparison:

    from runtime.ag_reduce_scatter import alltoallv, init_alltoallv

Variable-length AllToAllV public API:

    from runtime.trainium_alltoallv import all_to_allv, init_alltoallv
"""
from runtime.trainium_alltoallv import init_alltoallv, alltoallv, all_to_allv, compute_recv_counts
