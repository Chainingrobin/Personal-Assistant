import soundfile as sf
import numpy as np

audio, sr = sf.read("userclips/youssef/check calender.wav", dtype="int16", always_2d=True)
left = audio[:, 0]
right = audio[:, 1]
print(f"sr={sr}")
print(f"left  peak={np.abs(left).max()}  rms={np.sqrt(np.mean(left.astype(np.float64)**2)):.1f}")
print(f"right peak={np.abs(right).max()}  rms={np.sqrt(np.mean(right.astype(np.float64)**2)):.1f}")