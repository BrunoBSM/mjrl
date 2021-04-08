import time
from tqdm import tqdm

pbar = tqdm(unit=" transitions", desc="Transitions collected")

for i in range(100):
    pbar.update(10000)
    time.sleep(0.2)
