import os
import sys
import re
import traceback
import atexit
from typing import Generator, Union, Optional
from glob import glob

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response, Query
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from io import BytesIO
from pydantic import BaseModel
import threading

from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from call_llm import call_chat_complete, GPT4OMINI

i18n = I18nAuto()


# ============== LLM 情绪识别 ==============
def detect_emotion_with_llm(text: str, available_emotions: list, max_retries: int = 2) -> Optional[str]:
    """
    使用 LLM 根据文本内容识别情绪

    Args:
        text: 要合成的文本
        available_emotions: 该说话人可用的情绪列表
        max_retries: 最大重试次数

    Returns:
        识别出的情绪，如果识别失败返回 None
    """
    if not available_emotions:
        return None

    # 构建 system prompt
    emotions_str = "、".join(available_emotions)
    system_prompt = f"""你是一个情绪分析专家。你的任务是根据用户提供的文本，判断该文本最适合用哪种情绪来朗读。

可选的情绪有：{emotions_str}

规则：
1. 只能从上述可选情绪中选择一个
2. 只输出情绪名称，不要输出任何其他内容
3. 如果文本情绪不明显，优先选择"中立"或"默认"（如果可用）"""

    user_prompt = f"请分析以下文本应该用什么情绪朗读：\n\n{text}"

    for attempt in range(max_retries + 1):
        try:
            response = call_chat_complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=50,
                temperature=0.1,
                timeout=30,
                use_model=GPT4OMINI
            )

            # 清理响应
            emotion = response.strip().strip('"').strip("'").strip()

            # 验证响应是否在可用情绪列表中
            if emotion in available_emotions:
                print(f"[LLM] 识别情绪: '{emotion}' (文本: {text[:30]}...)")
                return emotion

            # 尝试模糊匹配
            for avail_emotion in available_emotions:
                if avail_emotion in emotion or emotion in avail_emotion:
                    print(f"[LLM] 模糊匹配情绪: '{avail_emotion}' (原始响应: {emotion})")
                    return avail_emotion

            print(f"[LLM] 第 {attempt + 1} 次尝试，响应 '{emotion}' 不在可用列表中")

        except Exception as e:
            print(f"[LLM] 第 {attempt + 1} 次调用失败: {e}")

    print(f"[LLM] 情绪识别失败，将使用默认情绪")
    return None
cut_method_names = get_cut_method_names()

# ============== 命令行参数 ==============
parser = argparse.ArgumentParser(description="Speaker-based GPT-SoVITS API")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="绑定地址, 默认 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default=9880, help="绑定端口, 默认 9880")
parser.add_argument("-s", "--speakers_dir", type=str, default="speakers", help="说话人目录, 默认 speakers")
args = parser.parse_args()

port = args.port
host = args.bind_addr
speakers_dir = os.path.join(now_dir, args.speakers_dir)
argv = sys.argv

# ============== frpc 反向代理 ==============
frpc_process = None

def start_frpc():
    """启动 frpc 反向代理"""
    global frpc_process
    frpc_dir = os.path.join(now_dir, "frpc")
    frpc_exe = os.path.join(frpc_dir, "frpc.exe")
    frpc_config = os.path.join(frpc_dir, "frpc.toml")

    if not os.path.exists(frpc_exe):
        print(f"[frpc] 警告: frpc.exe 不存在于 {frpc_exe}")
        return False

    if not os.path.exists(frpc_config):
        print(f"[frpc] 警告: frpc.toml 不存在于 {frpc_config}")
        return False

    try:
        frpc_process = subprocess.Popen(
            [frpc_exe, "-c", frpc_config],
            cwd=frpc_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        print(f"[frpc] 已启动反向代理 (PID: {frpc_process.pid})")
        return True
    except Exception as e:
        print(f"[frpc] 启动失败: {e}")
        return False

def stop_frpc():
    """停止 frpc 反向代理"""
    global frpc_process
    if frpc_process is not None:
        try:
            frpc_process.terminate()
            frpc_process.wait(timeout=5)
            print("[frpc] 已停止反向代理")
        except Exception as e:
            print(f"[frpc] 停止时出错: {e}")
            try:
                frpc_process.kill()
            except:
                pass
        frpc_process = None

atexit.register(stop_frpc)

# ============== 说话人管理 ==============
class SpeakerManager:
    def __init__(self, speakers_dir: str):
        self.speakers_dir = speakers_dir
        self.speakers_cache = {}
        self.current_speaker = None
        self.tts_pipeline = None
        self._scan_speakers()

    def _scan_speakers(self):
        """扫描所有说话人"""
        self.speakers_cache = {}
        if not os.path.exists(self.speakers_dir):
            print(f"[SpeakerManager] 说话人目录不存在: {self.speakers_dir}")
            return

        for speaker_name in os.listdir(self.speakers_dir):
            speaker_path = os.path.join(self.speakers_dir, speaker_name)
            if not os.path.isdir(speaker_path):
                continue

            speaker_info = self._parse_speaker(speaker_name, speaker_path)
            if speaker_info:
                self.speakers_cache[speaker_name] = speaker_info
                print(f"[SpeakerManager] 发现说话人: {speaker_name}, 情绪: {list(speaker_info['emotions'].keys())}")

    def _parse_speaker(self, speaker_name: str, speaker_path: str) -> Optional[dict]:
        """解析说话人信息"""
        # 查找模型文件
        gpt_files = glob(os.path.join(speaker_path, "*.ckpt"))
        sovits_files = glob(os.path.join(speaker_path, "*.pth"))

        if not gpt_files or not sovits_files:
            print(f"[SpeakerManager] 跳过 {speaker_name}: 缺少模型文件")
            return None

        # 使用第一个找到的模型
        gpt_path = gpt_files[0]
        sovits_path = sovits_files[0]

        # 查找情绪音频
        emotions = {}
        emotions_dir_pattern = os.path.join(speaker_path, "reference_audios", "*", "emotions")
        emotions_dirs = glob(emotions_dir_pattern)

        for emotions_dir in emotions_dirs:
            if not os.path.isdir(emotions_dir):
                continue

            for audio_file in os.listdir(emotions_dir):
                if not audio_file.endswith(".wav"):
                    continue

                # 解析情绪和 prompt: 【情绪】prompt.wav
                match = re.match(r"【(.+?)】(.+)\.wav$", audio_file)
                if match:
                    emotion = match.group(1)
                    prompt_text = match.group(2)
                    audio_path = os.path.join(emotions_dir, audio_file)

                    if emotion not in emotions:
                        emotions[emotion] = {
                            "audio_path": audio_path,
                            "prompt_text": prompt_text
                        }

        if not emotions:
            print(f"[SpeakerManager] 跳过 {speaker_name}: 没有找到情绪音频")
            return None

        # 检测语言 (从目录名推断)
        lang = "zh"
        if "-JP" in speaker_name or "_JP" in speaker_name:
            lang = "ja"
        elif "-EN" in speaker_name or "_EN" in speaker_name:
            lang = "en"

        return {
            "name": speaker_name,
            "gpt_path": gpt_path,
            "sovits_path": sovits_path,
            "emotions": emotions,
            "default_lang": lang
        }

    def get_speakers(self) -> list:
        """获取所有说话人列表"""
        return list(self.speakers_cache.keys())

    def get_emotions(self, speaker: str) -> list:
        """获取指定说话人的情绪列表"""
        if speaker not in self.speakers_cache:
            return []
        return list(self.speakers_cache[speaker]["emotions"].keys())

    def get_speaker_info(self, speaker: str) -> Optional[dict]:
        """获取说话人信息"""
        return self.speakers_cache.get(speaker)

    def _find_fallback_emotion(self, speaker_info: dict) -> Optional[str]:
        """查找默认情绪"""
        emotions = speaker_info["emotions"]
        # 优先使用 "中立"，然后 "默认"
        for fallback in ["中立", "默认"]:
            if fallback in emotions:
                return fallback
        # 都没有则返回第一个
        if emotions:
            return list(emotions.keys())[0]
        return None

    def load_speaker(self, speaker: str):
        """加载说话人模型"""
        if speaker not in self.speakers_cache:
            raise ValueError(f"说话人不存在: {speaker}")

        if self.current_speaker == speaker and self.tts_pipeline is not None:
            return  # 已加载

        speaker_info = self.speakers_cache[speaker]

        print(f"[SpeakerManager] 加载说话人: {speaker}")
        print(f"  GPT: {speaker_info['gpt_path']}")
        print(f"  SoVITS: {speaker_info['sovits_path']}")

        # 初始化 TTS pipeline
        if self.tts_pipeline is None:
            tts_config = TTS_Config("")
            self.tts_pipeline = TTS(tts_config)

        # 加载模型
        self.tts_pipeline.init_t2s_weights(speaker_info["gpt_path"])
        self.tts_pipeline.init_vits_weights(speaker_info["sovits_path"])

        self.current_speaker = speaker
        print(f"[SpeakerManager] 说话人 {speaker} 加载完成")

    def synthesize(self, speaker: str, emotion: str, text: str, text_lang: str,
                   speed_factor: float = 1.0, streaming_mode: Union[bool, int] = False,
                   **kwargs) -> Generator:
        """合成语音"""
        if speaker not in self.speakers_cache:
            raise ValueError(f"说话人不存在: {speaker}")

        speaker_info = self.speakers_cache[speaker]

        # 查找情绪
        if emotion not in speaker_info["emotions"]:
            fallback = self._find_fallback_emotion(speaker_info)
            if fallback:
                print(f"[SpeakerManager] 情绪 '{emotion}' 不存在，使用 '{fallback}'")
                emotion = fallback
            else:
                raise ValueError(f"说话人 {speaker} 没有可用的情绪")

        emotion_info = speaker_info["emotions"][emotion]

        # 加载模型
        self.load_speaker(speaker)

        # 确定 prompt 语言
        prompt_lang = speaker_info["default_lang"]

        # 构建请求
        req = {
            "text": text,
            "text_lang": text_lang.lower(),
            "ref_audio_path": emotion_info["audio_path"],
            "prompt_text": emotion_info["prompt_text"],
            "prompt_lang": prompt_lang,
            "top_k": kwargs.get("top_k", 5),
            "top_p": kwargs.get("top_p", 1),
            "temperature": kwargs.get("temperature", 1),
            "text_split_method": kwargs.get("text_split_method", "cut5"),
            "batch_size": kwargs.get("batch_size", 1),
            "batch_threshold": kwargs.get("batch_threshold", 0.75),
            "split_bucket": kwargs.get("split_bucket", True),
            "speed_factor": speed_factor,
            "fragment_interval": kwargs.get("fragment_interval", 0.3),
            "seed": kwargs.get("seed", -1),
            "parallel_infer": kwargs.get("parallel_infer", True),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.35),
            "sample_steps": kwargs.get("sample_steps", 32),
            "super_sampling": kwargs.get("super_sampling", False),
            "streaming_mode": False,
            "return_fragment": False,
        }

        # 处理 streaming_mode
        if streaming_mode == 0 or streaming_mode is False:
            req["streaming_mode"] = False
            req["return_fragment"] = False
        elif streaming_mode == 1 or streaming_mode is True:
            req["streaming_mode"] = False
            req["return_fragment"] = True
        elif streaming_mode == 2:
            req["streaming_mode"] = True
            req["return_fragment"] = False
        elif streaming_mode == 3:
            req["streaming_mode"] = True
            req["return_fragment"] = False
            req["fixed_length_chunk"] = True

        return self.tts_pipeline.run(req)


# ============== 初始化 ==============
speaker_manager = SpeakerManager(speakers_dir)

# ============== FastAPI ==============
APP = FastAPI(title="Speaker-based TTS API")

class TTSRequest(BaseModel):
    speaker: str
    emotion: Optional[str] = None  # None 或空字符串时自动识别
    text: str
    text_lang: str = "zh"
    speed_factor: float = 1.0
    streaming_mode: Union[bool, int] = False
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    seed: int = -1


def pack_wav(data: np.ndarray, rate: int) -> bytes:
    """打包为 WAV 格式"""
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    io_buffer.seek(0)
    return io_buffer.read()


def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    """生成 WAV 头"""
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()


@APP.get("/speakers")
async def get_speakers():
    """获取所有说话人列表"""
    speakers = speaker_manager.get_speakers()
    return JSONResponse(content={"speakers": speakers})


@APP.get("/emotions")
async def get_emotions(speaker: str = Query(..., description="说话人名称")):
    """获取指定说话人的情绪列表"""
    if speaker not in speaker_manager.speakers_cache:
        return JSONResponse(status_code=404, content={"message": f"说话人不存在: {speaker}"})

    emotions = speaker_manager.get_emotions(speaker)
    return JSONResponse(content={"speaker": speaker, "emotions": emotions})


@APP.get("/tts")
async def tts_get(
    speaker: str = Query(..., description="说话人名称"),
    emotion: Optional[str] = Query(None, description="情绪 (留空则自动识别)"),
    text: str = Query(..., description="要合成的文本"),
    text_lang: str = Query("zh", description="文本语言"),
    speed_factor: float = Query(1.0, description="语速"),
    streaming_mode: Union[bool, int] = Query(False, description="流式模式"),
):
    """TTS 推理 (GET)"""
    return await tts_handle(speaker, emotion, text, text_lang, speed_factor, streaming_mode)


@APP.post("/tts")
async def tts_post(request: TTSRequest):
    """TTS 推理 (POST)"""
    return await tts_handle(
        request.speaker,
        request.emotion,
        request.text,
        request.text_lang,
        request.speed_factor,
        request.streaming_mode,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
        text_split_method=request.text_split_method,
        batch_size=request.batch_size,
        seed=request.seed,
    )


async def tts_handle(speaker: str, emotion: Optional[str], text: str, text_lang: str,
                     speed_factor: float, streaming_mode: Union[bool, int], **kwargs):
    """TTS 处理"""
    # 参数验证
    if not speaker:
        return JSONResponse(status_code=400, content={"message": "speaker 是必需的"})
    if not text:
        return JSONResponse(status_code=400, content={"message": "text 是必需的"})
    if speaker not in speaker_manager.speakers_cache:
        return JSONResponse(status_code=404, content={"message": f"说话人不存在: {speaker}"})

    # 未指定情绪时，使用 LLM 自动识别
    if not emotion or emotion.strip() == "":
        available_emotions = speaker_manager.get_emotions(speaker)
        detected = detect_emotion_with_llm(text, available_emotions)
        if detected:
            emotion = detected
        else:
            # LLM 识别失败，使用回退逻辑
            emotion = speaker_manager._find_fallback_emotion(speaker_manager.speakers_cache[speaker])
            print(f"[TTS] LLM 识别失败，使用默认情绪: {emotion}")

    try:
        tts_generator = speaker_manager.synthesize(
            speaker=speaker,
            emotion=emotion,
            text=text,
            text_lang=text_lang,
            speed_factor=speed_factor,
            streaming_mode=streaming_mode,
            **kwargs
        )

        is_streaming = streaming_mode not in [0, False]

        if is_streaming:
            def streaming_generator():
                if_first_chunk = True
                for sr, chunk in tts_generator:
                    if if_first_chunk:
                        yield wave_header_chunk(sample_rate=sr)
                        if_first_chunk = False
                    yield chunk.tobytes()

            return StreamingResponse(streaming_generator(), media_type="audio/wav")
        else:
            sr, audio_data = next(tts_generator)
            audio_bytes = pack_wav(audio_data, sr)
            return Response(audio_bytes, media_type="audio/wav")

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "TTS 推理失败", "error": str(e)})


@APP.get("/")
async def root():
    """API 信息"""
    return {
        "name": "Speaker-based TTS API",
        "endpoints": {
            "/speakers": "GET - 获取所有说话人列表",
            "/emotions": "GET - 获取指定说话人的情绪列表",
            "/tts": "GET/POST - TTS 推理",
        }
    }


# ============== 主入口 ==============
if __name__ == "__main__":
    try:
        if host == "None":
            host = None

        # 启动 frpc 反向代理
        start_frpc()

        print(f"[API] 已扫描到 {len(speaker_manager.speakers_cache)} 个说话人")
        for speaker in speaker_manager.get_speakers():
            emotions = speaker_manager.get_emotions(speaker)
            print(f"  - {speaker}: {emotions}")

        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        stop_frpc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)