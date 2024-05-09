# ComfyUI-XTTS
a custom comfyui node for [coqui-ai/TTS](https://github.com/coqui-ai/TTS.git)'s xtts module! support 17 languages voice cloning and tts

 English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko) Hindi (hi)

<div>
  <figure>
  <img alt='webpage' src="web.png?raw=true" width="600px"/>
  <figure>
</div>

# Disclaimer  / 免责声明
We do not hold any responsibility for any illegal usage of the codebase. Please refer to your local laws about DMCA and other related laws.
我们不对代码库的任何非法使用承担任何责任. 请参阅您当地关于 DMCA (数字千年法案) 和其他相关法律法规.


## Features
- `srt` file for subtitle was supported
- mutiple speaker was supported in finetune and inference by `srt`
- huge comfyui custom nodes can merge in xtts

## How to use
make sure `ffmpeg` is worked in your commandline
for Linux
```
apt update
apt install ffmpeg
```
for Windows,you can install `ffmpeg` by [WingetUI](https://github.com/marticliment/WingetUI) automatically

then!
```
git clone https://github.com/AIFSH/ComfyUI-XTTS.git
cd ComfyUI-XTTS
pip install -r requirements.txt
```
`weights` will be downloaded from huggingface automatically! if you in china,make sure your internet attach the huggingface
or if you still struggle with huggingface, you may try follow [hf-mirror](https://hf-mirror.com/) to config your env.

或者下载[权重文件]()解压后把`pretrained_models`整个文件夹放进`ComfyUI-XTTS`目录

## Tutorial
[Demo](https://www.bilibili.com/video/BV1Wt421u7tu)

## WeChat Group && Donate
<div>
  <figure>
  <img alt='Wechat' src="wechat.jpg?raw=true" width="300px"/>
  <img alt='donate' src="donate.jpg?raw=true" width="300px"/>
  <figure>
</div>

## Thanks
[coqui-ai/TTS](https://github.com/coqui-ai/TTS.git)
