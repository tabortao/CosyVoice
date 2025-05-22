import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

def init_cosyvoice(model_path, load_jit=False, load_trt=False, fp16=False, use_flow_cache=False):
    """
    初始化 CosyVoice2 模型
    
    参数:
        model_path (str): 模型路径
        load_jit (bool): 是否使用JIT加载
        load_trt (bool): 是否使用TensorRT
        fp16 (bool): 是否使用FP16精度
        use_flow_cache (bool): 是否使用flow缓存
    
    返回:
        CosyVoice2: 初始化好的模型实例
    """
    return CosyVoice2(model_path, load_jit=load_jit, load_trt=load_trt, 
                      fp16=fp16, use_flow_cache=use_flow_cache)

def load_prompt_audio(audio_path, sample_rate):
    """
    加载提示音频文件
    
    参数:
        audio_path (str): 音频文件路径
        sample_rate (int): 采样率
    
    返回:
        tensor: 加载的音频数据
    """
    return load_wav(audio_path, sample_rate)

def generate_speech(model, text, style_text, prompt_speech, stream=False):
    """
    生成语音
    
    参数:
        model (CosyVoice2): CosyVoice2模型实例
        text (str): 要转换的文本
        style_text (str): 风格文本
        prompt_speech (tensor): 提示音频
        stream (bool): 是否使用流式生成
    
    返回:
        list: 生成的音频列表
    """
    results = []
    for i, result in enumerate(model.inference_zero_shot(text, style_text, prompt_speech, stream=stream)):
        output_path = f'zero_shot_{i}.wav'
        torchaudio.save(output_path, result['tts_speech'], model.sample_rate)
        results.append(output_path)
    return results

def main():
    """
    主函数，用于执行语音生成流程
    """
    # 初始化模型
    model = init_cosyvoice('pretrained_models/CosyVoice2-0.5B')
    
    # 加载提示音频
    prompt_speech = load_prompt_audio('./asset/zero_shot_prompt.wav', 16000)
    
    # 设置转换文本
    text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
    style_text = '希望你以后能够做的比我还好呦。'
    
    # 生成语音
    output_files = generate_speech(model, text, style_text, prompt_speech)
    print(f'生成的音频文件: {output_files}')

if __name__ == '__main__':
    main()
