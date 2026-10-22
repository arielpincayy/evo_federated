import re,sys
t=re.sub(r"[ \t]+"," ",open(sys.argv[1]).read()); pat=sys.argv[2]; n=int(sys.argv[3]); k=int(sys.argv[4]) if len(sys.argv)>4 else 0
ms=[m.start() for m in re.finditer(pat,t)]
print(f"[{len(ms)} matches]")
if ms: i=ms[min(k,len(ms)-1)]; print(re.sub(r"\n{2,}","\n",t[i:i+n]))
