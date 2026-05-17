"""Headless-render the poster.

Outputs:
    poster_full.png      — bitmap snapshot at 96 dpi, full size (for screen review)
    poster_top.png       — top half (title + workflow)
    poster_bot.png       — bottom half (results bands)
    poster_thumb.png     — downscaled for quick inspection
    poster.pdf           — print-ready PDF at the physical poster size
                            (3 ft × 4 ft = 914 mm × 1219 mm)
"""
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent.resolve()
HTML = HERE / "poster.html"

# Physical dimensions: 3 ft wide × 4 ft tall = 914 mm × 1219 mm.
# At 96 dpi (1 mm = 3.7795 px) that's 3455 × 4607 px for screenshot.
MM_W, MM_H = 914, 1219
PX_W = int(MM_W * 3.7795)
PX_H = int(MM_H * 3.7795)

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
    ctx = browser.new_context(viewport={"width": PX_W, "height": PX_H},
                               device_scale_factor=1.0)
    page = ctx.new_page()
    page.goto(f"file://{HTML}")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(500)

    # --- Bitmap previews -------------------------------------------------
    page.screenshot(path=str(HERE / "poster_full.png"),
                    full_page=False,
                    clip={"x": 0, "y": 0, "width": PX_W, "height": PX_H})
    page.screenshot(path=str(HERE / "poster_top.png"),
                    full_page=False,
                    clip={"x": 0, "y": 0,
                          "width": PX_W, "height": PX_H // 2})
    page.screenshot(path=str(HERE / "poster_bot.png"),
                    full_page=False,
                    clip={"x": 0, "y": PX_H // 2,
                          "width": PX_W, "height": PX_H // 2})

    # --- Print-ready PDF at physical poster size ------------------------
    # Use Playwright's pdf() which honors the @page CSS rule for size.
    page.emulate_media(media="print")
    page.pdf(path=str(HERE / "poster.pdf"),
             width=f"{MM_W}mm", height=f"{MM_H}mm",
             margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
             print_background=True,
             prefer_css_page_size=True)

    browser.close()

# Downscaled thumbnail for quick visual review
try:
    from PIL import Image
    img = Image.open(HERE / "poster_full.png")
    w, h = img.size
    target_w = 1200
    target_h = int(h * target_w / w)
    img.thumbnail((target_w, target_h))
    img.save(HERE / "poster_thumb.png")
    print(f"poster_full.png   {w}×{h}")
    print(f"poster.pdf        {MM_W}×{MM_H} mm  (3 ft × 4 ft portrait)")
    print(f"poster_thumb.png  {target_w}×{target_h}")
except ImportError:
    print(f"poster_full.png   {PX_W}×{PX_H}")
    print(f"poster.pdf        {MM_W}×{MM_H} mm")
