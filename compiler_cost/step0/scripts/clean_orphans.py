import glob, os, shutil, json
n=0
for d in glob.glob("/home/ubuntu/corpus/*/"):
    if os.path.exists(d+"spec.json") and not os.path.exists(d+"meta.json"):
        shutil.rmtree(d+"cache", ignore_errors=True)
        for f in ("spec.json","capture.log"):
            try: os.remove(d+f)
            except OSError: pass
        n+=1
print("orphans cleaned:", n, "metas:", len(glob.glob("/home/ubuntu/corpus/*/meta.json")))
