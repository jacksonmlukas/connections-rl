import json, struct, itertools
import numpy as np

def header(path):
    with open(path,'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def tensor(path, hdr, base, key):
    meta = hdr[key]; s, e = meta['data_offsets']
    with open(path,'rb') as f:
        f.seek(base + s)
        return np.frombuffer(f.read(e - s), dtype=np.float32).reshape(meta['shape']).astype(np.float64)

names = {'sft':'connections-rl-sft-7b.safetensors','seed0':'connections-rl-grpo-7b.safetensors',
         'seed1':'connections-rl-grpo-7b-seed1.safetensors','seed2':'connections-rl-grpo-7b-seed2.safetensors'}
H = {k:(v,)+header(v) for k,v in names.items()}   # k -> (path, hdr, base)
mods = sorted({k.rsplit('.lora_',1)[0] for k in H['sft'][1] if '.lora_' in k})
print(f'modules: {len(mods)}   (LoRA rank r=16, effective updates compared without materializing dense dW)')

def factors(key, m):
    p,h,b = H[key]
    return tensor(p,h,b,m+'.lora_B.weight'), tensor(p,h,b,m+'.lora_A.weight')

# dW_RL(seed) = B_g A_g - B_s A_s  == Bcat @ Acat  with rank 32
def rl_factors(seed, m):
    Bg,Ag = factors(seed,m); Bs,As = factors('sft',m)
    return np.concatenate([Bg,-Bs],axis=1), np.concatenate([Ag,As],axis=0)

def ip(B1,A1,B2,A2):   # <B1A1, B2A2>_F  via small matrices
    return float(np.trace((B1.T @ B2) @ (A2 @ A1.T)))

seeds = ['seed0','seed1','seed2']
acc_ip = {p:0.0 for p in itertools.combinations(seeds,2)}
acc_sq = {s:0.0 for s in seeds}
sft_sq = 0.0
ip_sft = {s:0.0 for s in seeds}
per_mod = []

for m in mods:
    RF = {s: rl_factors(s,m) for s in seeds}
    Bs,As = factors('sft',m)
    sft_sq += ip(Bs,As,Bs,As)
    for s in seeds:
        B,A = RF[s]
        acc_sq[s] += ip(B,A,B,A)
        ip_sft[s] += ip(B,A,Bs,As)
    for a,b in acc_ip:
        acc_ip[(a,b)] += ip(*RF[a], *RF[b])
    B0,A0 = RF['seed0']; B1,A1 = RF['seed1']
    n0 = ip(B0,A0,B0,A0)**0.5; n1 = ip(B1,A1,B1,A1)**0.5
    per_mod.append((m, ip(B0,A0,B1,A1)/(n0*n1), n0))

print('\n=== magnitude of RL-induced change (Frobenius over all modules) ===')
print(f"SFT update  (base -> SFT)   ||dW|| = {sft_sq**0.5:.4f}")
for s in seeds:
    print(f"{s}: RL update ||dW|| = {acc_sq[s]**0.5:.4f}   ({acc_sq[s]**0.5/sft_sq**0.5:.2f}x SFT update)")

print('\n=== cosine similarity between independent seeds RL-update directions ===')
for (a,b),v in acc_ip.items():
    print(f"cos({a}, {b}) = {v/((acc_sq[a]*acc_sq[b])**0.5):+.4f}")

print('\n=== control: cos(RL update, SFT update) ===')
for s in seeds:
    print(f"cos({s}_RL, SFT) = {ip_sft[s]/((acc_sq[s]*sft_sq)**0.5):+.4f}")

dim = sum(np.prod(H['sft'][1][m+'.lora_B.weight']['shape'][:1]) * H['sft'][1][m+'.lora_A.weight']['shape'][1] for m in mods)
print(f"\nparameter dim of dW space = {int(dim):,};  E[|cos|] for random directions ~ {np.sqrt(2/(np.pi*dim)):.2e}")

per_mod.sort(key=lambda t: -t[2])
print('\n=== per-module cos(seed0, seed1), 10 largest-change modules ===')
for m,c,n in per_mod[:10]:
    print(f"  {m.replace('base_model.model.model.layers.','L').replace('.weight',''):<34} cos={c:+.3f}  ||dW||={n:.4f}")
cs = np.array([c for _,c,_ in per_mod])
print(f"\nper-module cos(seed0,seed1): median={np.median(cs):+.3f}  mean={cs.mean():+.3f}  min={cs.min():+.3f}  max={cs.max():+.3f}")
