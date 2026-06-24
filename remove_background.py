"""
======================================================================
  PROFESSIONAL VIDEO BACKGROUND REMOVER
  Using: Robust Video Matting (RVM) — state-of-the-art model
         Handles hair, motion blur, fine edges, temporal consistency
======================================================================

SETUP (run once):
  pip install torch torchvision
  pip install opencv-python-headless
  pip install av                  # for WebM/MOV output

MODEL DOWNLOAD (automatic on first run, ~130MB):
  The script downloads the RVM MobileNetV3 model from GitHub.
  For best quality use the ResNet50 model (uncomment below).

HOW TO RUN:
  python remove_background.py

OUTPUT:
  - output_alpha.webm   → WebM with transparency (VP9 + alpha)
  - output_matte.mp4    → Grayscale alpha matte (debug view)
  - output_composite.mp4 → Green-screen composite (preview)
======================================================================
"""

import os
import sys
import urllib.request
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────
#  CONFIG — Edit these paths
# ──────────────────────────────────────────────
INPUT_VIDEO   = "hello.mp4"   # ← your video
OUTPUT_WEBM   = "output_alpha.webm"     # transparent WebM (use in browser/editors)
OUTPUT_MATTE  = "output_matte.mp4"      # alpha matte debug view
OUTPUT_GREEN  = "output_greenscreen.mp4"# green composite preview

# Model choice: "mobilenetv3" (fast) or "resnet50" (highest quality)
MODEL_VARIANT = "mobilenetv3"

# Downsample ratio — lower = faster but less detail (0.25–1.0)
# Use 0.4 for speed, 1.0 for maximum quality
DOWNSAMPLE_RATIO = 0.4

# Background color for composite preview (B, G, R)
BG_COLOR = (0, 255, 0)   # Green screen

# ──────────────────────────────────────────────
#  MODEL URLS
# ──────────────────────────────────────────────
MODEL_URLS = {
    "mobilenetv3": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth",
    "resnet50":    "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth",
}
MODEL_PATH = f"rvm_{MODEL_VARIANT}.pth"


# ──────────────────────────────────────────────
#  DOWNLOAD MODEL
# ──────────────────────────────────────────────
def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"✓ Model already downloaded: {MODEL_PATH}")
        return
    url = MODEL_URLS[MODEL_VARIANT]
    print(f"⬇  Downloading model from:\n   {url}")
    print("   This is ~130MB (mobilenetv3) or ~280MB (resnet50)...")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(downloaded / total_size * 100, 100)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r   [{bar}] {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, MODEL_PATH, reporthook=progress)
    print(f"\n✓ Model saved to: {MODEL_PATH}")


# ──────────────────────────────────────────────
#  LOAD MODEL  (inline RVM architecture)
# ──────────────────────────────────────────────
def load_model():
    """
    Load Robust Video Matting model.
    The model class is imported directly from a local clone or
    installed package. If you have the RVM repo cloned, add it to sys.path.
    
    ALTERNATIVE (recommended):
        git clone https://github.com/PeterL1n/RobustVideoMatting
        cd RobustVideoMatting
        python inference.py --variant mobilenetv3 \
            --checkpoint rvm_mobilenetv3.pth \
            --device cuda \
            --input-source ../input.mp4 \
            --output-type video \
            --output-composition ../output_alpha.mov \
            --output-alpha ../output_matte.mp4 \
            --downsample-ratio 0.4
    """
    try:
        # Try importing from local RobustVideoMatting clone
        sys.path.insert(0, "./RobustVideoMatting")
        from model import MattingNetwork
        print("✓ Loaded RVM from local clone (RobustVideoMatting/model.py)")
    except ImportError:
        try:
            # Try pip-installed package
            from robust_video_matting import MattingNetwork
            print("✓ Loaded RVM from pip package")
        except ImportError:
            print("\n" + "="*60)
            print("  MODEL NOT FOUND — Choose one setup method:")
            print("="*60)
            print("""
  OPTION A — Clone the repo (recommended):
    git clone https://github.com/PeterL1n/RobustVideoMatting
    python remove_background.py

  OPTION B — Use the repo's inference script directly:
    git clone https://github.com/PeterL1n/RobustVideoMatting
    cd RobustVideoMatting
    pip install -r requirements.txt
    python inference.py \\
      --variant mobilenetv3 \\
      --checkpoint ../rvm_mobilenetv3.pth \\
      --device cuda \\
      --input-source "../WhatsApp_Video_2026-03-21_at_15_24_59.mp4" \\
      --output-type video \\
      --output-composition ../output_alpha.mov \\
      --output-alpha ../output_matte.mp4 \\
      --downsample-ratio 0.4

  OPTION C — Use rembg (frame-by-frame, slower but simpler):
    pip install rembg[gpu]
    python -c "
    import cv2, rembg, numpy as np
    cap = cv2.VideoCapture('input.mp4')
    # See rembg_fallback() below
    "
""")
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device} {'(GPU ✓)' if device=='cuda' else '(CPU — may be slow)'}")

    model = MattingNetwork(MODEL_VARIANT).eval().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✓ Weights loaded from: {MODEL_PATH}")
    return model, device


# ──────────────────────────────────────────────
#  MAIN INFERENCE LOOP
# ──────────────────────────────────────────────
def process_video(model, device):
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"✗ Could not open: {INPUT_VIDEO}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n📹 Input:  {width}x{height} @ {fps:.2f}fps  ({total} frames)")

    fourcc_mp4  = cv2.VideoWriter_fourcc(*"mp4v")
    out_matte   = cv2.VideoWriter(OUTPUT_MATTE,  fourcc_mp4, fps, (width, height))
    out_green   = cv2.VideoWriter(OUTPUT_GREEN,  fourcc_mp4, fps, (width, height))

    # For WebM with alpha we use ffmpeg subprocess after collecting frames
    alpha_frames = []

    rec  = [None] * 4   # RVM recurrent state
    bgr  = torch.tensor(BG_COLOR, dtype=torch.float32, device=device).div(255)
    bgr  = bgr.view(1, 3, 1, 1)

    print(f"\n🔄 Processing {total} frames...\n")

    frame_idx = 0
    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Convert BGR → RGB tensor [1, 3, H, W] float32 0-1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            src = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0)
            src = src.float().div(255).to(device)

            # RVM forward pass
            fgr, pha, *rec = model(src, *rec, DOWNSAMPLE_RATIO)

            # Alpha matte (grayscale)
            alpha_np = pha[0, 0].cpu().numpy()
            alpha_u8 = (alpha_np * 255).clip(0, 255).astype(np.uint8)

            # Foreground (color-corrected)
            fgr_np = fgr[0].permute(1, 2, 0).cpu().numpy()
            fgr_u8 = (fgr_np * 255).clip(0, 255).astype(np.uint8)
            fgr_bgr = cv2.cvtColor(fgr_u8, cv2.COLOR_RGB2BGR)

            # Composite on green background
            alpha_3ch = np.stack([alpha_np] * 3, axis=-1)
            bg = np.full_like(fgr_np, [c/255 for c in BG_COLOR[::-1]])
            composite = (fgr_np * alpha_3ch + bg * (1 - alpha_3ch))
            composite_u8 = (composite * 255).clip(0, 255).astype(np.uint8)
            composite_bgr = cv2.cvtColor(composite_u8, cv2.COLOR_RGB2BGR)

            # Write outputs
            out_matte.write(cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2BGR))
            out_green.write(composite_bgr)

            # Store RGBA frame for WebM
            rgba = np.dstack([fgr_u8, alpha_u8])  # [H, W, 4]
            alpha_frames.append(rgba)

            frame_idx += 1
            if frame_idx % 10 == 0 or frame_idx == total:
                bar = "█" * int(frame_idx/total*40) + "░" * (40 - int(frame_idx/total*40))
                print(f"\r  [{bar}] {frame_idx}/{total}", end="", flush=True)

    cap.release()
    out_matte.release()
    out_green.release()
    print(f"\n\n✓ Matte:      {OUTPUT_MATTE}")
    print(f"✓ Composite:  {OUTPUT_GREEN}")

    return alpha_frames, fps, width, height


# ──────────────────────────────────────────────
#  EXPORT TRANSPARENT WEBM  (via ffmpeg)
# ──────────────────────────────────────────────
def export_transparent_webm(alpha_frames, fps, width, height):
    """
    Write individual RGBA PNGs then encode to WebM with alpha via ffmpeg.
    Requires: ffmpeg with libvpx-vp9 support (standard in most installs)
    """
    import subprocess, tempfile, shutil

    print("\n📦 Encoding transparent WebM...")
    tmpdir = tempfile.mkdtemp(prefix="rvm_frames_")

    try:
        for i, frame in enumerate(alpha_frames):
            path = os.path.join(tmpdir, f"frame_{i:06d}.png")
            # OpenCV needs BGRA
            bgra = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(path, bgra, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%06d.png"),
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-b:v", "0",
            "-crf", "18",
            "-auto-alt-ref", "0",
            OUTPUT_WEBM
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ffmpeg error: {result.stderr[-300:]}")
        else:
            size_mb = os.path.getsize(OUTPUT_WEBM) / 1e6
            print(f"✓ Transparent WebM: {OUTPUT_WEBM} ({size_mb:.1f} MB)")
    finally:
        shutil.rmtree(tmpdir)


# ──────────────────────────────────────────────
#  REMBG FALLBACK  (no RVM needed, frame-by-frame)
# ──────────────────────────────────────────────
def rembg_fallback():
    """
    Fallback using rembg — simpler, no repo clone needed.
    Quality is good for static content, less temporal consistency.
    
    Run:  pip install rembg[gpu] onnxruntime-gpu
    Then: python remove_background.py --rembg
    """
    try:
        from rembg import remove, new_session
    except ImportError:
        print("Install rembg: pip install rembg[gpu]")
        sys.exit(1)

    session = new_session("u2net")   # or "u2net_human_seg" for people
    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    alpha_frames = []

    import tempfile, subprocess, shutil
    tmpdir = tempfile.mkdtemp()

    print(f"🔄 rembg: processing {total} frames...")
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        _, buf = cv2.imencode(".png", frame)
        result = remove(buf.tobytes(), session=session, alpha_matting=True,
                        alpha_matting_foreground_threshold=240,
                        alpha_matting_background_threshold=10,
                        alpha_matting_erode_size=10)
        nparr = np.frombuffer(result, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)  # BGRA
        path = os.path.join(tmpdir, f"frame_{i:06d}.png")
        cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        alpha_frames.append(img)
        i += 1
        if i % 5 == 0:
            print(f"\r  {i}/{total}", end="", flush=True)

    cap.release()
    print(f"\n✓ Frames processed: {i}")

    # Encode WebM
    cmd = ["ffmpeg", "-y", "-framerate", str(fps),
           "-i", os.path.join(tmpdir, "frame_%06d.png"),
           "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
           "-b:v", "0", "-crf", "18", "-auto-alt-ref", "0", OUTPUT_WEBM]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmpdir)
    print(f"✓ Output: {OUTPUT_WEBM}")


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if "--rembg" in sys.argv:
        rembg_fallback()
    else:
        print("="*60)
        print("  Robust Video Matting — Professional Background Removal")
        print("="*60)
        download_model()
        model, device = load_model()
        frames, fps, w, h = process_video(model, device)
        export_transparent_webm(frames, fps, w, h)
        print("\n🎉 Done! Files saved:")
        print(f"   {OUTPUT_WEBM}   ← import into Premiere/DaVinci/After Effects")
        print(f"   {OUTPUT_MATTE} ← alpha matte for manual refinement")
        print(f"   {OUTPUT_GREEN} ← green-screen preview")
