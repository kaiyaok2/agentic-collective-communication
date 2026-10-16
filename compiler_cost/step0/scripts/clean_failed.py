import json, glob, os, shutil
n=0
for m in glob.glob("/home/ubuntu/corpus/*/meta.json"):
    d=json.load(open(m))
    if not d.get("ok") and "rc=-9" in str(d.get("error","")):
        dd=os.path.dirname(m); shutil.rmtree(os.path.join(dd,"cache"), ignore_errors=True)
        for f in ("meta.json","spec.json","capture.log"):
            try: os.remove(os.path.join(dd,f))
            except OSError: pass
        n+=1
print("cleaned", n, "killed captures; remaining metas:", len(glob.glob("/home/ubuntu/corpus/*/meta.json")))
