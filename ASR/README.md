# Improved Latvian ASR Model for Difficult Acoustic Conditions (Far-field, telephone etc)

## Model

The model is based on Encoder-Decoder architecture and uses Wav2Vec 2.0 XLS-R 300M as feature extractor.

```
    Text (BPE units)
      ^^^^^^^^^^^
-------------------------
|  Transformer decoder  |
-------------------------
            ^                        
            |
-------------------------
|  Conformer encoder    |
-------------------------
            ^                        
            |
-------------------------
|  Linear pre-encoder   |
-------------------------
            ^                        
            |
-------------------------
|  Wav2Vec2 XLS-R 300M  |
-------------------------
    ^^^^^^^^^^^^^^^
     Raw PCM Audio 
```

The model input is raw PCM audio which is processed by Wav2Vec 2.0 and passed to Conformer encoder via small linear pre-encoder layer. Finally, the Transformer-based decoder processes the output of the encoder and produces speech transcript as BPE subword units. 

The training is performed on 100h [Latvian Speech Recognition Corpus](https://aclanthology.org/L14-1257/), which is augmented for additional robustness:
- 3x speed perturbation (90%, 100% and 110%);
- 2x reverberation with simulated impulse responses;
- random low-pass filtering to emulate 8kHz recordings.

During the training first layers of Wav2Vec2 model are frozen, but all other layers are fine-tuned together with other parts of the model.

## Installation

Running the model requires working EspNet installation. 

Please follow the EspNet official documentation for installation: https://espnet.github.io/espnet/installation.html

The ASR model is not public and is available only upon request. Contact us at: tilde@tilde.com

You can download the model like this:
```bash
# exact link is provided upon request
wget https://asreeblobs.blob.core.windows.net/pip3models/xlsr_lv.zip?
```

## Demo

Below is the example code in Python demonstrating how to use the model. 


```python
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
import soundfile
from argparse import ArgumentParser

parser = ArgumentParser(description='ASR demo')
parser.add_argument(
        'file', type=str, help='audio file')
args = parser.parse_args()

d = ModelDownloader()
# It may takes a while to download and build models
speech2text = Speech2Text(
    **d.download_and_unpack("xlsr_lv.zip"),
    device="cuda",
    minlenratio=0.0,
    maxlenratio=0.0,
    ctc_weight=0.3,
    lm_weight=0.3,
    beam_size=20,
    batch_size=0,
    penalty=0.1,
    nbest=1
)
fs=16000

speech, rate = soundfile.read(args.file)
assert rate == fs, "models supports only {fs} sampling rate"
nbests = speech2text(speech)
text, *_ = nbests[0]

print(f"Input Speech: {args.file}")
print(f"ASR hypothesis: {text}")
print("*" * 50)
```

You run the script as follows:
```bash
python3 test.py path/to/audio_file.wav
```

Where "path/to/audio_file.wav" is as path to 16kHz WAV or FLAC file (see table of supported formats: http://www.mega-nerd.com/libsndfile/).

The output will look like something like:
```
Input Speech: /home/askars/cv_audio/lv/clips_wav/common_voice_lv_20788350.wav
ASR hypothesis: ko darīsim ar zēnu
**************************************************
```

You might need to activate EspNet environment first. These can be done as follows:
```bash
cd 'your_espnet_root_dir'
. tools/activate_python.sh
```