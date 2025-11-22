# Huggingface Implementation of AV-HuBERT on the MuAViC Dataset

![lip-reading](https://github.com/facebookresearch/av_hubert/blob/main/assets/lipreading.gif?raw=true)

This repository contains a Huggingface implementation of the AV-HuBERT (Audio-Visual Hidden Unit BERT) model, specifically trained and tested on the MuAViC (Multilingual Audio-Visual Corpus) dataset. AV-HuBERT is a self-supervised model designed for audio-visual speech recognition, leveraging both audio and visual modalities to achieve robust performance, especially in noisy environments.


### Requirements

```sh
git clone -b Petar https://github.com/nguyenvulebinh/AV-HuBERT-S2S.git
cd AV-HuBERT-S2S
conda create -n avhuberts2s python=3.9
conda activate avhuberts2s
pip install -r requirements.txt
python run_example.py
```

### Testing model

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

### Pre-train an AV-HuBERT model

Suppose `{train,valid}.tsv` are saved at `/path/to/data`, `{train,valid}.km`
are saved at `/path/to/labels`, the configuration file is saved at `/path/to/conf/conf-name`, and the label rate is 100Hz.

To train a model, run:
```sh
$ cd avhubert
$ fairseq-hydra-train --config-dir /path/to/conf/ --config-name conf-name \
  task.data=/path/to/data task.label_dir=/path/to/label \
  model.label_rate=100 hydra.run.dir=/path/to/experiment/pretrain/ \
  common.user_dir=`pwd`
```

### Finetune an AV-HuBERT model with Seq2Seq
Suppose `{train,valid}.tsv` are saved at `/path/to/data`, `{train,valid}.wrd`
are saved at `/path/to/labels`, the configuration file is saved at `/path/to/conf/conf-name`.

To fine-tune a pre-trained HuBERT model at `/path/to/checkpoint`, run:
```sh
$ cd avhubert
$ fairseq-hydra-train --config-dir /path/to/conf/ --config-name conf-name \
  task.data=/path/to/data task.label_dir=/path/to/label \
  task.tokenizer_bpe_model=/path/to/tokenizer model.w2v_path=/path/to/checkpoint \
  hydra.run.dir=/path/to/experiment/finetune/ common.user_dir=`pwd`
```

### Decode an AV-HuBERT model
Suppose the `test.tsv` and `test.wrd` are the video list and transcripts of
the split to be decoded, saved at `/path/to/data`, and the fine-tuned model is
saved at `/path/to/checkpoint`.

#### Seq2Seq decoding

`task.normalize` needs to be consistent with the value used during fine-tuning.
Decoding results will be saved at
`/path/to/experiment/decode/s2s/test`.

```sh
$ cd avhubert
$ python -B infer_s2s.py --config-dir ./conf/ --config-name conf-name \
  dataset.gen_subset=test common_eval.path=/path/to/checkpoint \
  common_eval.results_path=/path/to/experiment/decode/s2s/test \
  override.modalities=['video'] common.user_dir=`pwd`
```
#### Test under noisy environment
If you want to test your model under noisy environment, append the following to the above command.

`+override.noise_wav=/path/to/noise override.noise_prob=1 override.noise_snr={snr}` 

 `{snr}` is the signal-to-noise ratio (SNR) and `/path/to/noise` is a folder containing noise manifest files (`/path/to/noise/{valid,test}.tsv`). See [`preparation`](avhubert/preparation/) for setting up this folder.