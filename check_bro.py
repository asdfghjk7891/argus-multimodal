import sys
sys.path.insert(0, 'C:/Users/user/Desktop/Argus')
import loaders.load_picodomain as pico
from loaders.tdata import TData

pico.PICO_FOLDER = 'C:/Users/user/Desktop/Argus/data/picodomain/'
data = pico.load_pico_dist(start=0, end=57600, delta=3600,
                            is_test=False, use_flows=False)

print(f'is_test: {data.is_test}')
print(f'masks[0] dtype: {data.masks[0].dtype}')
print(f'masks[0]: {data.masks[0]}')
print(f'masks[0] shape: {data.masks[0].shape}')
print()

for i in range(3):
    ei = data.eis[i]
    mask = data.masks[i]
    print(f'슬라이스 {i}:')
    print(f'  ei shape: {ei.shape}')
    print(f'  mask shape: {mask.shape}')
    print(f'  mask sum: {mask.sum()}')
    print(f'  ~mask sum: {(~mask).sum()}')
    print(f'  TRAIN ei: {ei[:, mask].shape}')
    print(f'  VAL ei:   {ei[:, ~mask].shape}')