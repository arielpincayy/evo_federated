import re,sys
t=re.sub(r"\s+"," ",open(sys.argv[1]).read())
for pat in sys.argv[3:]:
    ms=list(re.finditer(pat,t,re.I)); print(f"\n== /{pat}/ {len(ms)} hits")
    for m in ms[:int(sys.argv[2])]: print("  …"+t[max(0,m.start()-200):m.end()+260]+"…")
