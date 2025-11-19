import os
import csv
import torch
from transformers import Speech2TextTokenizer
from src.model.avhubert2text import AV2TextForConditionalGeneration
from src.dataset.load_data import load_feature


# ---------------------------------------------------------
# PATHS – CHANGE THESE
# ---------------------------------------------------------

VIDEO_ROOT = r"C:\Pera\faks\ucenje_iz_podataka\mvlrs_v1\video\test"
AUDIO_ROOT = r"C:\Pera\faks\ucenje_iz_podataka\mvlrs_v1\audio\test"

WRD_FILE = r"C:\Pera\faks\ucenje_iz_podataka\mvlrs_v1\fixed_tsv\test.wrd"  
# <-- YOUR 1 FILE WITH MANY LINES


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

language = "en"
model_name_or_path = f"nguyenvulebinh/AV-HuBERT-MuAViC-{language}"

model = AV2TextForConditionalGeneration.from_pretrained(
    model_name_or_path, cache_dir="./model-bin"
).cuda().eval()

tokenizer = Speech2TextTokenizer.from_pretrained(
    model_name_or_path, cache_dir="./model-bin"
)


# ---------------------------------------------------------
# LOAD GROUND TRUTH LINES
# ---------------------------------------------------------

with open(WRD_FILE, "r", encoding="utf-8") as f:
    ground_truth_lines = [line.strip() for line in f.readlines()]

print(f"Loaded {len(ground_truth_lines)} ground-truth lines.")


# ---------------------------------------------------------
# SCAN AND SORT ALL VIDEO & AUDIO FILES
# ---------------------------------------------------------

def collect_sorted_files(root, ext):
    files = []
    for r, d, fs in os.walk(root):
        for f in fs:
            if f.endswith(ext):
                files.append(os.path.join(r, f))
    return sorted(files)  # IMPORTANT: sorted order must match ordering of WRD lines


video_files = collect_sorted_files(VIDEO_ROOT, ".mp4")
audio_files = collect_sorted_files(AUDIO_ROOT, ".wav")

print(f"Found {len(video_files)} videos and {len(audio_files)} audios.\n")

if len(video_files) != len(audio_files):
    raise ValueError("Video and audio counts do NOT match!")

if len(video_files) != len(ground_truth_lines):
    raise ValueError(
        f"Number of ground truth lines ({len(ground_truth_lines)}) "
        f"does not match number of video/audio samples ({len(video_files)})."
    )


# ---------------------------------------------------------
# RESULT LISTS
# ---------------------------------------------------------

predicted_list = []
ground_truth_list = []


# ---------------------------------------------------------
# PROCESS ALL SAMPLES
# ---------------------------------------------------------

for idx, (video_path, audio_path, gt_text) in enumerate(zip(video_files, audio_files, ground_truth_lines)):

    print(f"[{idx+1}/{len(video_files)}] Processing:")
    print(f"  Video: {video_path}")
    print(f"  Audio: {audio_path}")

    sample = load_feature(video_path, audio_path)

    audio_feats = sample["audio_source"].cuda()
    video_feats = sample["video_source"].cuda()

    # Attention mask
    attention_mask = torch.BoolTensor(audio_feats.size(0), audio_feats.size(-1)) \
                        .fill_(False).cuda()

    # Generate text
    output = model.generate(
        audio_feats,
        attention_mask=attention_mask,
        video=video_feats,
        max_length=1024,
    )

    pred_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    ground_truth_list.append(gt_text)
    predicted_list.append(pred_text)

    print("  GT:", gt_text)
    print("  PR:", pred_text)
    print("-" * 80)


# ---------------------------------------------------------
# SAVE CSV FILES
# ---------------------------------------------------------

with open("ground_truth.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ground_truth"])
    for line in ground_truth_list:
        writer.writerow([line])

with open("predictions.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["prediction"])
    for line in predicted_list:
        writer.writerow([line])

print("✔ DONE — Saved ground_truth.csv and predictions.csv")
