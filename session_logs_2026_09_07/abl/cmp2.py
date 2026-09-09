import re, os, json, glob
base="/home/ubuntu/abl"
probs=sorted([f[:-4] for f in os.listdir(base+"/base_r24") if f.endswith(".log")])
def cur_win(p):
    j=base+"/base_r24/"+p+"/kiss_summary.json"
    if os.path.exists(j):
        d=json.load(open(j)); return (d.get("best_name"), d.get("best_sim_time_us"))
    return (None,None)
def strat_win(p):
    # strat logs carry Winner/SimTime lines
    lf=base+"/strat_raw/"+p+".log"
    if os.path.exists(lf):
        txt=open(lf,errors="ignore").read()
        ws=re.findall(r"Winner:\s*(.+)",txt); sims=re.findall(r"SimTime:\s*([\d.]+)",txt)
        if ws and sims: return (ws[-1].strip(), float(sims[-1]))
    # fallback json
    j=base+"/strat_raw/"+p+"/kiss_summary.json"
    if os.path.exists(j):
        d=json.load(open(j)); return (d.get("best_name"), d.get("best_sim_time_us"))
    return (None,None)
print("%-40s %9s %9s %7s  verdict" % ("problem","CUR","STRAT","ratio"))
div=0; res=0; miss=[]
for p in probs:
    cw,cs=cur_win(p); sw,ss=strat_win(p)
    if cs is None or ss is None:
        miss.append((p,cs,ss,cw,sw)); print("%-40s %9s %9s %7s  MISSING" % (p,str(cs),str(ss),"?")); continue
    res+=1; r=ss/cs
    if r>1.05: v="DIVERGE(strat worse)"; div+=1
    elif r<0.95: v="STRAT-BETTER"
    else: v="match"
    print("%-40s %9.1f %9.1f %6.2fx  %s" % (p,cs,ss,r,v))
print("\nresolved=%d  diverge=%d" % (res,div))
for m in miss: print("  MISS",m)
