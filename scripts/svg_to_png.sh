#!/usr/bin/env bash
# Rasterize media/*.svg to media/*.png via headless Chromium (playwright's,
# works on Linux boxes with no system Chrome), then autocrop with Pillow.
# Retina-crisp via device_scale_factor=2.
set -euo pipefail
cd "$(dirname "$0")/../media"

uv run --no-project --with playwright --with pillow python - <<'EOF'
import glob
import subprocess
import sys

from playwright.sync_api import sync_playwright

subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"],
               check=True, capture_output=True)

from PIL import Image, ImageChops  # noqa: E402

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 2400},
                            device_scale_factor=2)
    for svg in sorted(glob.glob("*.svg")):
        png = svg[:-4] + ".png"
        page.goto(f"file://{__import__('os').getcwd()}/{svg}")
        page.wait_for_timeout(1500)  # let the cdnjs webfonts load
        page.screenshot(path=png)
        img = Image.open(png).convert("RGB")
        bg = Image.new("RGB", img.size,
                       img.getpixel((img.width - 1, img.height - 1)))
        bbox = ImageChops.difference(img, bg).getbbox()
        if bbox:
            pad = 24
            bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                    min(img.width, bbox[2] + pad),
                    min(img.height, bbox[3] + pad))
            img.crop(bbox).save(png)
        print(png, Image.open(png).size)
    browser.close()
EOF
