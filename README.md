# Huggingface Implementation of AV-HuBERT on the MuAViC Dataset

![lip-reading](https://github.com/facebookresearch/av_hubert/blob/main/assets/lipreading.gif?raw=true)

This repository contains a Huggingface implementation of the AV-HuBERT (Audio-Visual Hidden Unit BERT) model, specifically trained and tested on the MuAViC (Multilingual Audio-Visual Corpus) dataset. AV-HuBERT is a self-supervised model designed for audio-visual speech recognition, leveraging both audio and visual modalities to achieve robust performance, especially in noisy environments.


### Inference code

```sh
git clone https://github.com/nguyenvulebinh/AV-HuBERT-S2S.git
cd AV-HuBERT-S2S
conda create -n avhuberts2s python=3.9
conda activate avhuberts2s
pip install -r requirements.txt
python run_example.py
```

```python
from src.model.avhubert2text import AV2TextForConditionalGeneration
from src.dataset.load_data import load_feature
from transformers import Speech2TextTokenizer
import torch

if __name__ == "__main__":
    # Choose language to run example
    AVAILABEL_LANGUAGES = ["en"]
    language = "en"
    assert language in AVAILABEL_LANGUAGES, f"Language {language} is not available, please choose one of {AVAILABEL_LANGUAGES}"
    
    
    # Load model and tokenizer
    model_name_or_path = f"nguyenvulebinh/AV-HuBERT-MuAViC-{language}"
    model = AV2TextForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir='./model-bin')
    tokenizer = Speech2TextTokenizer.from_pretrained(model_name_or_path, cache_dir='./model-bin')
    
    model = model.cuda().eval()
    
    # Load example video and audio
    video_example = f"./example/video_processed/{language}_lip_movement.mp4"
    audio_example = f"./example/video_processed/{language}_audio.wav"
    if not os.path.exists(video_example) or not os.path.exists(audio_example):
        print(f"WARNING: Example video and audio for {language} is not available english will be used instead")
        video_example = f"./example/video_processed/en_lip_movement.mp4"
        audio_example = f"./example/video_processed/en_audio.wav"
    
    # Load and process example
    sample = load_feature(
        video_example,
        audio_example
    )
    
    audio_feats = sample['audio_source'].cuda()
    video_feats = sample['video_source'].cuda()
    attention_mask = torch.BoolTensor(audio_feats.size(0), audio_feats.size(-1)).fill_(False).cuda()
    
    # Generate text
    output = model.generate(
        audio_feats,
        attention_mask=attention_mask,
        video=video_feats,
        max_length=1024,
    )

    print(tokenizer.batch_decode(output, skip_special_tokens=True))
```

### Data preprocessing scripts

```sh
mkdir model-bin
cd model-bin
wget https://huggingface.co/nguyenvulebinh/AV-HuBERT/resolve/main/20words_mean_face.npy .
wget https://huggingface.co/nguyenvulebinh/AV-HuBERT/resolve/main/shape_predictor_68_face_landmarks.dat .

python src/dataset/video_to_audio_lips.py
```

### Pretrained AVSR model

<table align="center">
    <tr>
        <th>Languages</th>
        <th>Huggingface</th>
    </tr>
    <tr>
        <th>English</th>
        <th><a href="https://huggingface.co/nguyenvulebinh/AV-HuBERT-MuAViC-en">Checkpoint-EN</a></th>
    </tr>
</table>
