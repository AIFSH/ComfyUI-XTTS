import site
import os,sys
import logging
from server import PromptServer

now_dir = os.path.dirname(os.path.abspath(__file__))
site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if(site_packages_roots==[]):site_packages_roots=["%s/runtime/Lib/site-packages" % now_dir]
#os.environ["OPENBLAS_NUM_THREADS"] = "4"
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/XTTS.pth" % (site_packages_root), "w") as f:
                f.write(
                    "%s\n%s/TTS\n"
                    % (now_dir,now_dir)
                )
            break
        except PermissionError:
            raise PermissionError

if os.path.isfile("%s/XTTS.pth" % (site_packages_root)):
    print("!!!XTTS path was added to " + "%s/XTTS.pth" % (site_packages_root) 
    + "\n if meet `No module` error,try `python main.py` again")


WEB_DIRECTORY = "./web"
from .nodes import LoadSRT,LoadAudioPath, PreViewAudio,XTTS_INFER, XTTS_INFER_SRT

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadAudioPath": LoadAudioPath,
    "PreViewAudio": PreViewAudio,
    "LoadSRT": LoadSRT,
    "XTTS_INFER": XTTS_INFER,
    "XTTS_INFER_SRT": XTTS_INFER_SRT
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudioPath": "LoadAudioPath",
    "PreViewAudio": "PreView Audio",
    "LoadSRT": "SRT FILE Loader",
    "XTTS_INFER": "XTTS Inference",
    "XTTS_INFER_SRT": "XTTS Inference with srt"
}

@PromptServer.instance.routes.get("/xtts/reboot")
def restart(self):
    try:
        sys.stdout.close_log()
    except Exception as e:
        pass

    return os.execv(sys.executable, [sys.executable] + sys.argv)
