wget --no-check-certificate "https://onedrive.live.com/download?cid=EDF068524DEBC79F&resid=EDF068524DEBC79F%211077&authkey=ACO8aUAM6pV8jmw" -O "checkpoints.zip"
unzip checkpoints.zip
cp -f checkpoints/swin_large_patch4_window12_384_22kto1k.pth model_zoo/swin
