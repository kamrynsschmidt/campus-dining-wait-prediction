"""Rebuilds every result and figure in the paper, in order. Usage: python run_all.py
Takes about 10–20 minutes on a 2-core CPU. Outputs go to results/ and figures/."""
import subprocess, sys, time
steps = ["run_experiments.py", "run_e5.py", "run_e6.py", "make_figures.py"]
t0 = time.time()
for s in steps:
    print(f"\n=== {s} ===", flush=True)
    subprocess.run([sys.executable, s], check=True)
print(f"\nAll steps finished in {(time.time() - t0) / 60:.1f} minutes.")
