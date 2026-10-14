import sys; sys.path.insert(0, "/home/ubuntu/OverlayCCL")
from search.problems import get_problem
import search.problems_modext
for p, b in (("tp_mlp", "per_mb"), ("fsdp_prefetch", "per_mb"), ("pp_send_recv", "per_mb"), ("llama_block_ar", "sequential_2ar")):
    t = get_problem(p).builtin_templates[b]; print(f"##### {p}/{b}  mark_step calls: {t.count('mark_step')}"); print(t[:800])
