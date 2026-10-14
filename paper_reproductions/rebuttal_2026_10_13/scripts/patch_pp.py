import re
p='/home/ubuntu/OverlayCCL/training/train_pp_send_recv_7node.py'; s=open(p).read()
old="        flat_in = torch.stack(acts, dim=0)\n"
new="        flat_in = torch.stack(acts, dim=0)\n        if not flat_in.requires_grad:\n            flat_in.requires_grad_()\n"
assert old in s and new not in s; open(p,'w').write(s.replace(old,new,1)); print("pp harness patched")
