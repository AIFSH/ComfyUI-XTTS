import os
import time
import torch
import torchaudio
import folder_paths
from huggingface_hub import snapshot_download
from .TTS.tts.models.xtts import Xtts
from .TTS.tts.configs.xtts_config import XttsConfig

now_dir = os.path.dirname(os.path.abspath(__file__))
input_path = folder_paths.get_input_directory()
output_path = folder_paths.get_output_directory()
pretrained_models_path = os.path.join(now_dir, "pretrained_models")
os.makedirs(pretrained_models_path, exist_ok=True)

class XTTS_INFER:
    @classmethod
    def INPUT_TYPES(s):
        language_list = ["en","es","fr","de","it","pt","pl","tr","ru",
                         "nl","cs","ar","zh-cn","ja","hu","ko","hi"]
        return {"required":
                    {"audio": ("AUDIO",),
                     "text": ("STRING",{
                         "default": "你好啊！世界",
                         "multiline": True
                      }),
                     "language": (language_list,{
                         "default": "zh-cn"
                     }),
                     "temperature":("FLOAT",{
                         "default":0.65,
                         "min":0,
                         "max": 1,
                         "step": 0.05,
                         "display": "slider"
                     }),
                     "length_penalty":("FLOAT",{
                         "default":1.0
                     }),
                     "repetition_penalty":("FLOAT",{
                         "default":2.0
                     }),
                     "top_k":("INT",{
                         "default":50,
                     }),
                     "top_p":("FLOAT",{
                         "default":0.8,
                         "min":0,
                         "max": 1,
                         "step": 0.05,
                         "display": "slider"
                     }),
                     "speed":("FLOAT",{
                         "default":1.0,
                     }),
                     }
                }

    CATEGORY = "AIFSH_XTTS"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ("AUDIO",)

    OUTPUT_NODE = False

    FUNCTION = "get_wav_tts"

    def get_wav_tts(self, audio, text,language,temperature,length_penalty,
                    repetition_penalty,top_k,top_p,speed):
        if not os.path.isfile(os.path.join(pretrained_models_path,"model.pth")):
            print("Downloading model...")
            snapshot_download(repo_id="coqui/XTTS-v2",revision="v2.0.3",local_dir=pretrained_models_path)
        print("Loading model...")
        config = XttsConfig()
        config.load_json(os.path.join(pretrained_models_path, "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=pretrained_models_path, use_deepspeed=True)
        model.cuda()
        
        print("Computing speaker latents...")
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[audio])

        print("Inference...")
        out = model.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=temperature, # Add custom parameters here
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            speed=speed,
            enable_text_splitting=True
        )
        wav_path = os.path.join(output_path, f"{time.time()}_xtts.wav")
        torchaudio.save(wav_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
        return (wav_path,)

class PreViewAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"audio": ("AUDIO",),}
                }

    CATEGORY = "AIFSH_XTTS"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_name = os.path.basename(audio)
        tmp_path = os.path.dirname(audio)
        audio_root = os.path.basename(tmp_path)
        return {"ui": {"audio":[audio_name,audio_root]}}

class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["wav", "mp3","WAV","flac","m4a"]]
        return {"required":
                    {"audio": (sorted(files),)},
                }

    CATEGORY = "AIFSH_XTTS"

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        return (audio_path,)

class LoadSRT:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["srt", "txt"]]
        return {"required":
                    {"srt": (sorted(files),)},
                }

    CATEGORY = "AIFSH_XTTS"

    RETURN_TYPES = ("SRT",)
    FUNCTION = "load_srt"

    def load_srt(self, srt):
        srt_path = folder_paths.get_annotated_filepath(srt)
        return (srt_path,)