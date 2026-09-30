from pathlib import Path
import re, subprocess, math
nums = [1,2,3,4,21,22,23,24,25]
for n in nums:
    p = next(Path("lecture_notes").glob(f"lecture_{n:02d}_*.md"))
    s = p.read_text(encoding="utf-8")
    old = subprocess.check_output(["git", "show", "HEAD:" + p.as_posix()]).decode("utf-8")
    assert re.findall(r"^\*\*Transcript coverage:\*\*.*$", s, re.M) == re.findall(r"^\*\*Transcript coverage:\*\*.*$", old, re.M), p
    assert s.split("---",2)[1] == old.split("---",2)[1], p
    assert not re.search(r"^(<<<<<<<|=======|>>>>>>>)", s, re.M), p
    fences = re.findall(r"^\s*\$\$\s*$", s, re.M)
    assert len(fences) % 2 == 0, p
    for block in re.findall(r"(?ms)^\s*\$\$\s*$\n(.*?)^\s*\$\$\s*$", s):
        assert block.count("{") == block.count("}"), (p, block)
    assert "\ufffd" not in s, p
    # Each section keeps its explanation layer; no source coverage is removed.
    assert s.count("### Additional explanation") == old.count("### Additional explanation"), p
print("Nine notes: metadata, transcript coverage, explanation sections, math fences/braces, and text integrity verified.")
# Small mathematical checks independent of the prose.
for H in [1,2,10,100]:
    for eps in [0,0.001,0.1,1]:
        exact = sum(1-(1-eps)**t for t in range(1,H+1))
        upper = min(H,eps*H*(H+1)/2)
        assert exact <= upper+1e-10
for gamma in [0.1,0.5,0.9]:
    # Alternating two-state chain: normalized discounted occupancy from state 0.
    mu = [1/(1+gamma),gamma/(1+gamma)]
    r = [2,-1]
    value = (2-gamma)/(1-gamma**2)
    assert abs(sum(a*b for a,b in zip(mu,r))/(1-gamma)-value)<1e-10
for lam in [0,0.5,0.95,1]:
    for K in [1,2,10]:
        assert abs(sum((1-lam)*lam**(n-1) for n in range(1,K))+lam**(K-1)-1)<1e-12
for K in [2,5,10]:
    eta=0.1
    b=math.log(eta/(K-1))
    a=math.log(1-eta)-b
    assert a>0 and abs(a+b-math.log(1-eta))<1e-12
print("Finite-horizon bound, successor normalization, finite GAE weights, and softened goal-reward identity checked.")
