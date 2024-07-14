<h1 align="center">
  <br>
  SimpleSlowVideoUpscaler
  <br>
</h1>

<h3 align="center">Video version of goKay's Tile Upscaler, made with Claude and patience</h3>

<div align="center">

[![Twitter Follow](https://img.shields.io/twitter/follow/hrishi?style=social)](https://twitter.com/hrishioa)

</div>

<div align="center">

https://github.com/user-attachments/assets/e80f6d43-119e-4bf2-a886-79eac885e624

</div>

I was super impressed with the output from [tile upscaler](https://huggingface.co/spaces/gokaygokay/Tile-Upscaler), and the outputs were stable enough to use for video. If you take time tuning the settings, you can get a pretty good output - except that the model (or the LoRA) wants to give everyone eye makeup.

**This is just meant to be a simple script so I could run some videos through. Not for production usage!**

I haven't been able to fully get it working on M1 Macs (with the ANE). Any help appreciated!

# Installation

First install cuda for your python if not already present:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install the requirements:

```bash
pip install -r requirements.py
```

Install ffmpeg - [get it here for Windows](https://www.ffmpeg.org/download.html), or for POSIX:

```bash
sudo apt install ffmpeg
```

Run the upscaler!

```bash
python upscaler.py
```

# Usage

![CleanShot 2024-07-14 at 11 47 30@2x](https://github.com/user-attachments/assets/b0cb7b76-eba3-4535-99fb-c0a6a130fcd6)

You can abort processing at any time. Max frames to process, and skipping frames is super helpful if you just want to check the output. The defaults are sensible ones I found work in general. Play with the resolution - some videos work better at higher res (even though they themselves are lower res) and sometimes otherwise!

# Thanks

Built with claude-sonnet on a Saturday, building on the work of:

- [nonda30 or gokaygokay](https://x.com/nonda30)
- [philz1337x](https://x.com/philz1337x)
- [fermat_research](https://x.com/fermat_research)
