import os, json, glob
base="/home/ubuntu/abl"
def sum_tokens(arm):
    calls=inp=out=0
    for tj in glob.glob(base+"/"+arm+"/*/tokens.jsonl"):
        for ln in open(tj,errors="ignore"):
            ln=ln.strip()
            if not ln: continue
            try: d=json.loads(ln)
            except: continue
            calls+=1
            inp+=d.get("input_tokens",0) or d.get("in",0) or 0
            out+=d.get("output_tokens",0) or d.get("out",0) or 0
    return calls,inp,out
for arm in ["base_r24","strat_raw"]:
    c,i,o=sum_tokens(arm)
    # sonnet-4-5 pricing $3/$15 per 1M
    cost=i/1e6*3 + o/1e6*15
    print("%-12s calls=%4d in=%9d out=%9d  $%.2f" % (arm,c,i,o,cost))
