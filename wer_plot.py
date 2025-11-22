import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# ---- CONFIG ----
noise_audio_folder = r"C:\Pera\faks\AV-HuBERT-S2S\audio_noise"
clean_csv_path = r"C:\Pera\faks\AV-HuBERT-S2S\wer_cer_results.csv"
noise_csv_path = r"C:\Pera\faks\AV-HuBERT-S2S\wer_cer_results_noise.csv"

# Load CSVs
df_clean = pd.read_csv(clean_csv_path)
df_noise = pd.read_csv(noise_csv_path)

# Extract SNR from filenames (supports negative SNR)
snr_map = {}  # index → snr

for fname in os.listdir(noise_audio_folder):
    # REGEX UPDATED: supports -5, -10, -20
    match = re.search(r"_(\d+)_SNR(-?\d+)", fname)
    if match:
        index = int(match.group(1))
        snr = int(match.group(2))
        snr_map[index] = snr

# Map SNR to noise CSV rows
df_noise["snr"] = df_noise["index"].map(snr_map)

# Compute average clean WER
clean_wer = df_clean["wer"].mean() * 100

# Compute average WER per SNR
noise_grouped = (df_noise.groupby("snr")["wer"].mean() * 100).sort_index()

# X-axis categories
x_labels = ["clear"] + [f"SNR{snr}" for snr in noise_grouped.index]
y_values = [clean_wer] + list(noise_grouped.values)

# Plot
plt.figure(figsize=(12, 6))
plt.bar(x_labels, y_values)

plt.ylabel("WER (%)")
plt.title("WER for Clean Audio and Each SNR Level")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("wer_plot_separate_snrs.png", dpi=200)
plt.show()

print("Plot saved as wer_plot_separate_snrs.png")
