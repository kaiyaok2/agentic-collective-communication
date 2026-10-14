"""B>1 fix for training/train_olmoe10b.py's MoEBlock: replace the fancy-index gather (xf[token_idx]) and index_add scatter,
which the Neuron runtime executes as indirect-DMA (vector DGE) and faults with NRT_EXEC_OOB once T=B*S>256, by one-hot
matmuls (dense, no indirect copy). Also derive the expert-choice capacity from T (tokens per step), not SEQLEN.
Apply with: python3 olmoe_b_fix.py <in> <out> [BSZ] [DM]
"""
import re, sys
src = open(sys.argv[1]).read(); out = sys.argv[2]; bsz = int(sys.argv[3]) if len(sys.argv) > 3 else 2; dm = int(sys.argv[4]) if len(sys.argv) > 4 else None
old_fwd = src[src.index("    def forward(self, x):\n        B, S, D = x.shape"):src.index("        return out.reshape(B, S, D)\n") + len("        return out.reshape(B, S, D)\n")]
new_fwd = '''    def forward(self, x):
        B, S, D = x.shape
        T = B * S
        xf = x.reshape(T, D)
        gp = F.softmax(self.gate(xf), dim=-1)              # (T, NEXP=ws)
        gp_t = gp.transpose(0, 1).contiguous()              # (ws, T)
        scores, token_idx = gp_t.topk(self.cap, dim=1)      # each (ws, cap)
        # One-hot selection matrix (ws*cap, T): dense matmul gather/scatter instead of indirect-DMA fancy indexing,
        # which faults (NRT_EXEC_OOB) on this runtime once T > 256.
        onehot = F.one_hot(token_idx.reshape(-1), T).to(xf.dtype)   # (ws*cap, T)
        sel = (onehot @ xf).view(self.ws, self.cap, D)               # == xf[token_idx]
        send = sel.reshape(-1)
        xm.mark_step()
        recv = _A2AV.apply(send, self.mc, self.a2av_fn)     # (ws*cap*D,)
        proc = self.expert(recv.view(self.ws * self.cap, D))
        xm.mark_step()
        combined = _A2AV.apply(proc.reshape(-1), self.mc, self.a2av_fn)
        weighted = combined.view(self.ws, self.cap, D) * scores.unsqueeze(-1)
        out = onehot.transpose(0, 1) @ weighted.reshape(-1, D)        # == zeros(T,D).index_add(0, token_idx, weighted)
        return out.reshape(B, S, D)
'''
src = src.replace(old_fwd, new_fwd)
src = re.sub(r"^BSZ *=.*$", f"BSZ       = {bsz}", src, flags=re.M)
if dm: src = re.sub(r"^DM *=.*$", f"DM        = {dm}", src, flags=re.M)
# capacity from tokens per step (T = BSZ*SEQLEN), not SEQLEN
src = src.replace("cap = max(1, int(math.ceil(math.ceil(SEQLEN * TOPK / ws) * EXPANSION)))", "cap = max(1, int(math.ceil(math.ceil(BSZ * SEQLEN * TOPK / ws) * EXPANSION)))")
src = src.replace("times[200:]", "times[100:]").replace("len(times) - 200", "len(times) - 100").replace("len(times) > 200", "len(times) > 100").replace("from step 200", "from step 100")
open(out, "w").write(src); print("wrote", out, "BSZ", bsz, "DM", dm)
