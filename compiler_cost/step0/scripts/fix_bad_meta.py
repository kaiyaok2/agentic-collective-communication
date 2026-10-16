import glob, json, os, shutil
bad=0
for m in glob.glob("/home/ubuntu/corpus/*/meta.json"):
    try: json.load(open(m))
    except Exception:
        d=os.path.dirname(m); shutil.rmtree(d+"/cache", ignore_errors=True)
        for f in ("meta.json","spec.json","capture.log"):
            try: os.remove(os.path.join(d,f))
            except OSError: pass
        bad+=1; print("removed corrupt", d)
print("corrupt metas removed:", bad)
