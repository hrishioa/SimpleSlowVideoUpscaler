<h1 align="center">
  <br>
  SimpleSlowVideoUpscaler
  <br>
</h1>

<h3 align="center">Video version of [goKay's Tile Upscaler](https://huggingface.co/spaces/gokaygokay/Tile-Upscaler), made with Claude and patience</h3>

<div align="center">

[![Twitter Follow](https://img.shields.io/twitter/follow/hrishi?style=social)](https://twitter.com/hrishioa) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

<div align="center">

![output2](https://github.com/hrishioa/wishful-search/assets/973967/34e2fa82-2ae2-442a-972d-a2ab97d51d5e)

</div>

Was super impressed with the output from [tile upscaler](https://huggingface.co/spaces/gokaygokay/Tile-Upscaler), and the outputs were stable enough to use for video. If you take time tuning the settings, you can get a pretty good output - except that the model (or the LoRA) wants to give everyone eye makeup.

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

[Image of program]

You can abort processing at any time. Max frames to process, and skipping frames is super helpful if you just want to check the output. The defaults are sensible ones I found work in general. Play with the resolution - some videos work better at higher res (even though they themselves are lower res) and sometimes otherwise!

# Thanks

Built with claude-sonnet on a Saturday, building on the work of:

- [nonda30 or gokaygokay](https://x.com/nonda30)
- [philz1337x](https://x.com/philz1337x)
- [fermat_research](https://x.com/fermat_research)
