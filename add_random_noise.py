import os
import random
import numpy as np
import soundfile as sf
import librosa

NOISE_DIR = r"C:\Pera\faks\AV-HuBERT-S2S\example\noise_samples"
INPUT_DIR = r"C:\Pera\faks\ucenje_iz_podataka\mvlrs_v1\audio\test"
OUTPUT_DIR = r"C:\Pera\faks\AV-HuBERT-S2S\audio_noise"
SAMPLE_RATE = 16000
SNR_LIST = [-20, -10, -5, 5, 10, 20]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_audio(path, sr=16000):
    """Load WAV into proper range [-32768, 32767]."""
    audio, orig_sr = librosa.load(path, sr=sr)
    audio = (audio * 32767).astype(np.int16)
    return audio


def add_noise(signal, noise, snr):
    signal = signal.astype(np.float32)
    noise = noise.astype(np.float32)

    if len(noise) < len(signal):
        repeats = int(np.ceil(len(signal) / len(noise)))
        noise = np.tile(noise, repeats)

    noise = noise[:len(signal)]

    amp_s = np.sqrt(np.mean(signal**2))
    amp_n = np.sqrt(np.mean(noise**2))

    if amp_n < 1e-6:  # noise is silent
        return signal.astype(np.int16)

    scale = (amp_s / amp_n) / (10 ** (snr / 20))
    noise *= scale

    mixed = signal + noise
    mixed = np.clip(mixed, -32768, 32767)

    return mixed.astype(np.int16)


def get_noise_files():
    noise_files = []
    for root, dirs, files in os.walk(NOISE_DIR):
        for f in files:
            if f.lower().endswith(".wav"):
                noise_files.append(os.path.join(root, f))
    return noise_files


noise_files = get_noise_files()
print("Found noise files:", len(noise_files))


# 🔥 RECURSIVE SEARCH FOR WAV FILES 🔥
wav_files = []
for root, dirs, files in os.walk(INPUT_DIR):
    for f in files:
        if f.lower().endswith(".wav"):
            wav_files.append(os.path.join(root, f))

print("Found input audio files:", len(wav_files))


for audio_path in wav_files:
    signal = load_audio(audio_path, SAMPLE_RATE)

    noise_file = random.choice(noise_files)
    noise = load_audio(noise_file, SAMPLE_RATE)

    snr = random.choice(SNR_LIST)

    mixed = add_noise(signal, noise, snr)

    # Create clean output filename
    folder_name = os.path.basename(os.path.dirname(audio_path))
    base = os.path.splitext(os.path.basename(audio_path))[0]
    outname = f"{folder_name}_{base}_SNR{snr}.wav"

    outpath = os.path.join(OUTPUT_DIR, outname)

    sf.write(outpath, mixed, SAMPLE_RATE)
    print("Saved:", outpath)
