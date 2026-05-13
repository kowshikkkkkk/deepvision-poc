import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

SIGNAL_CLASSES = [
    "pure_sine", "noisy_sine", "square_wave", "sawtooth_wave", "chirp",
    "am_modulated", "fm_modulated", "triangle_wave", "random_noise", "mixed_harmonics"
]

SAMPLE_RATE = 8000
DURATION    = 1.0
N_SAMPLES   = int(SAMPLE_RATE * DURATION)

def _generate_signal(class_id, seed):
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    f0  = rng.uniform(200, 1200)
    if class_id == 0:   s = np.sin(2 * np.pi * f0 * t)
    elif class_id == 1: s = np.sin(2 * np.pi * f0 * t) + rng.normal(0, 0.3, N_SAMPLES)
    elif class_id == 2: s = np.sign(np.sin(2 * np.pi * f0 * t))
    elif class_id == 3: s = 2 * (t * f0 - np.floor(0.5 + t * f0))
    elif class_id == 4:
        f1 = rng.uniform(f0*2, f0*4)
        s  = np.sin(2 * np.pi * (f0*t + 0.5*((f1-f0)/DURATION)*t**2))
    elif class_id == 5:
        fc = rng.uniform(800, 2000)
        s  = (1 + 0.5*np.sin(2*np.pi*f0*t)) * np.sin(2*np.pi*fc*t)
    elif class_id == 6:
        s = np.sin(2*np.pi*f0*t + rng.uniform(1,5)*np.sin(2*np.pi*(f0/10)*t))
    elif class_id == 7: s = 2*np.abs(2*(t*f0 - np.floor(t*f0+0.5))) - 1
    elif class_id == 8: s = rng.normal(0, 1, N_SAMPLES)
    else:
        freqs = [f0, f0*2, f0*3, f0*5]
        amps  = [1.0, 0.5, 0.25, 0.125]
        s     = sum(a*np.sin(2*np.pi*f*t) for f,a in zip(freqs, amps))
    mx = np.abs(s).max()
    return s / (mx + 1e-8)

def _signal_to_image(signal, img_size=64):
    n_fft = 256; hop = 64
    window = np.hanning(n_fft)
    frames = []
    for start in range(0, len(signal)-n_fft, hop):
        frame = signal[start:start+n_fft] * window
        frames.append(np.abs(np.fft.rfft(frame)))
    spec    = np.array(frames).T
    log_spec = 20 * np.log10(spec + 1e-8)
    lo, hi  = log_spec.min(), log_spec.max()
    img_arr = ((log_spec - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    img = Image.fromarray(img_arr, mode="L").resize((img_size, img_size)).convert("RGB")
    return img

class SyntheticSignalDataset(Dataset):
    def __init__(self, n_per_class=500, img_size=64, transform=None, seed_offset=0):
        self.n_per_class = n_per_class
        self.img_size    = img_size
        self.transform   = transform
        self.seed_offset = seed_offset
        self.index = [(cls, i) for cls in range(len(SIGNAL_CLASSES)) for i in range(n_per_class)]

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        class_id, sample_i = self.index[idx]
        seed   = self.seed_offset + class_id * 100_000 + sample_i
        signal = _generate_signal(class_id, seed)
        img    = _signal_to_image(signal, self.img_size)
        if self.transform: img = self.transform(img)
        return img, class_id

def get_signal_loaders(n_train=400, n_val=50, n_test=50,
                       img_size=64, batch_size=64, num_workers=0):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    train_set = SyntheticSignalDataset(n_train, img_size, transform, seed_offset=0)
    val_set   = SyntheticSignalDataset(n_val,   img_size, transform, seed_offset=1_000_000)
    test_set  = SyntheticSignalDataset(n_test,  img_size, transform, seed_offset=2_000_000)
    print(f"[Signal] Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    return (DataLoader(train_set, batch_size, shuffle=True,  num_workers=num_workers),
            DataLoader(val_set,   batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(test_set,  batch_size, shuffle=False, num_workers=num_workers))