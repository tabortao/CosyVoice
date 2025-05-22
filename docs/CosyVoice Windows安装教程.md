# CosyVoice Windows安装教程

> CosyVoice2.0是阿里巴巴通义实验室推出的CosyVoice语音生成大模型升级版，模型用有限标量量化技术提高码本利用率，简化文本-语音语言模型架构，推出块感知因果流匹配模型支持多样的合成场景。CosyVoice 2在发音准确性、音色一致性、韵律和音质上都有显著提升，MOS评测分从5.4提升到5.53，支持流式推理，大幅降低首包合成延迟至150ms，适合实时语音合成场景。

项目地址：https://github.com/FunAudioLLM/CosyVoice

## 克隆和安装
```bash
# 克隆项目
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive

# 创建虚拟环境，不要新建py310目录，使用相对路径创建
micromamba create -p ./py310 python=3.10
# 激活虚拟环境，这里主要需要为相对路径的环境
micromamba activate ./py310

# 安装依赖
# 使用micromamba 安装pynini，其实是调用了conda进行了安装
micromamba install -y -c conda-forge pynini==2.1.5

# 注意关掉代理，否则会走代理，浪费很多流量。

# 安装其他依赖,前面使用了相对路径创建了虚拟环境，这里需要指定下python路径
./py310/python.exe -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

F:\Code\TTS\CosyVoice\py310\python.exe -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

```

## 模型下载

### 方式一：SDK模型下载(推荐)
新建`download_model.py`文件，复制下面代码到文件中，放到下面根目录。

```python
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```
执行如下命令开始下载：
```bash
 ./py310/python.exe download_model.py
```

### 方式二：Git模型下载

```bash
# git模型下载，请确保已安装git lfs
mkdir -p pretrained_models
git lfs install
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-300M.git pretrained_models/CosyVoice-300M
git clone https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git pretrained_models/CosyVoice-300M-SFT
git clone https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git pretrained_models/CosyVoice-300M-Instruct
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```
Optionally, you can unzip ttsfrd resouce and install ttsfrd package for better text normalization performance.

Notice that this step is not necessary. If you do not install ttsfrd package, we will use WeTextProcessing by default.

```bash
cd pretrained_models/CosyVoice-ttsfrd/
# Windows采用这个命令解压
Expand-Archive -Path "resource.zip" -DestinationPath "." 
# unzip resource.zip -d .
F:\Code\TTS\CosyVoice\py310\python.exe -m pip install ttsfrd_dependency-0.1-py3-none-any.whl
# pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

## 用法

### 基本用法
强烈建议使用`CosyVoice2-0.5B`模型。
```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
```
### CosyVoice2用法
```bash
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# save zero_shot spk for future usage
assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech_16k, 'my_zero_shot_spk') is True
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
cosyvoice.save_spkinfo()

# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
    torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# instruct usage
for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# bistream usage, you can use generator as input, this is useful when using text llm model as input
# NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
def text_generator():
    yield '收到好友从远方寄来的生日礼物，'
    yield '那份意外的惊喜与深深的祝福'
    yield '让我心中充满了甜蜜的快乐，'
    yield '笑容如花儿般绽放。'
for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```

### Web Demo用法
```bash
# change iic/CosyVoice-300M-SFT for sft inference, or iic/CosyVoice-300M-Instruct for instruct inference
.\py310\python.exe webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```

F:\Code\TTS\CosyVoice\py310\python.exe webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B