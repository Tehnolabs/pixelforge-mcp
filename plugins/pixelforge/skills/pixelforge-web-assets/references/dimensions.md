# Pixel dimensions and crop recipes for PixelForge web assets

Read this when the target size is not one of the three defaults in SKILL.md (a
1200x630 card, a 16:9 or 21:9 hero, a square icon), or when a crop must be computed
for a source whose measured size differs from the table.

## Contents

1. Native output sizes per aspect ratio and tier
2. Crop formula for any target
3. Recipes for common web targets
4. Icon file set, tags and manifest
5. Safe zones

## 1. Native output sizes per aspect ratio and tier

Google's published output dimensions for `gemini-nano-banana-2.1` and
`gemini-3.1-flash-image`. `gemini-3-pro-image` matches them for the ten standard
ratios; `gemini-3.1-flash-lite-image` outputs the 1K column only. The values are
rounded to multiples of 16, so the ratios are approximate (16:9 is really 1.79 to 1).
Google can change the table and the model can deviate from it, so measure the file
(`file PATH` prints width x height) before computing a crop.

| aspect_ratio | 1K | 2K | 4K |
|---|---|---|---|
| 1:1 | 1024x1024 | 2048x2048 | 4096x4096 |
| 3:2 | 1264x848 | 2528x1696 | 5056x3392 |
| 2:3 | 848x1264 | 1696x2528 | 3392x5056 |
| 4:3 | 1200x896 | 2400x1792 | 4800x3584 |
| 3:4 | 896x1200 | 1792x2400 | 3584x4800 |
| 5:4 | 1152x928 | 2304x1856 | 4608x3712 |
| 4:5 | 928x1152 | 1856x2304 | 3712x4608 |
| 16:9 | 1376x768 | 2752x1536 | 5504x3072 |
| 9:16 | 768x1376 | 1536x2752 | 3072x5504 |
| 21:9 | 1584x672 | 3168x1344 | 6336x2688 |
| 4:1 (panoramic) | 2048x512 | 4096x1024 | 8192x2048 |
| 1:4 (panoramic) | 512x2048 | 1024x4096 | 2048x8192 |
| 8:1 (panoramic) | 3072x384 | 6144x768 | 12288x1536 |
| 1:8 (panoramic) | 384x3072 | 768x6144 | 1536x12288 |

- `0.5K` exists on `gemini-3.1-flash-image` only and halves the 1K column (16:9
  gives 688x384).
- The panoramic rows render only on `gemini-nano-banana-2.1` and
  `gemini-3.1-flash-image`. Other models return a standard shape without an error.
- `image_size` is one of `0.5K`, `1K`, `2K`, `4K`; `generate_image` rejects a size
  the chosen model cannot output. `quality="balanced"` fixes 1K and
  `quality="quality"` fixes 2K; `quality="fast"` is 1K only.

Tier choice: the smallest row that covers the target on both edges after the crop.
1K covers everything up to about 1376 wide (cards, README heroes, icons, banners
from the panoramic rows); 2K covers landing heroes, YouTube banners and 1080-wide
social posts; 4K is for print.

## 2. Crop formula for any target

Target W x H with ratio R = W / H. Source w x h measured from the file.

```
if w / h > R:            # source is wider than the target
    cw = round(h * R); x = (w - cw) // 2
    transform_image(operation="crop", x=x, y=0, width=cw, height=h)
else:                    # source is taller than the target
    ch = round(w / R); y = (h - ch) // 2
    transform_image(operation="crop", x=0, y=y, width=w, height=ch)
transform_image(operation="resize", width=W, height=H, maintain_aspect=false)
```

The resize is a downscale with under 1 percent stretch because the crop already
matches R. Never resize upward: `maintain_aspect=true` will not enlarge at all, and
`maintain_aspect=false` enlarges with blur. If the source is smaller than the target
on either edge, regenerate at the next tier instead.

To keep an off-centre subject, move the crop window. Call
`detect_objects(image_path=..., objects="the main subject")`; each detection carries
`box_2d` as `[y_min, x_min, y_max, x_max]` on a 0-1000 scale. Convert with
`px = value * w / 1000` (or `* h / 1000` for y) and choose `x` or `y` so the box
stays inside the kept region.

## 3. Recipes for common web targets

Sources are at 1K on the default model unless the tier column says 2K. Crop values
assume the measured size equals section 1; recompute with the formula when it does
not. `resize` uses `maintain_aspect=false` unless noted.

| Target | Pixels | Source ratio, tier | crop (x, y, width, height) | resize |
|---|---|---|---|---|
| Open Graph, Twitter large card, LinkedIn share, blog cover | 1200x630 | 16:9, 1K | 0, 23, 1376, 722 | 1200x630 |
| GitHub repository social preview | 1280x640 | 16:9, 1K | 0, 40, 1376, 688 | 1280x640 |
| YouTube thumbnail | 1280x720 | 16:9, 1K | none (5, 0, 1365, 768 for exact 16:9) | 1280x720 |
| YouTube channel banner | 2560x1440 | 16:9, 2K | none | 2560x1440 |
| X (Twitter) header | 1500x500 | 4:1, 1K | 256, 0, 1536, 512 | 1500x500 |
| X header on a non-panoramic model | 1500x500 | 21:9, 1K | 0, 72, 1584, 528 | 1500x500 |
| Email header | 600x200 | as X header | as X header | 600x200 |
| LinkedIn personal banner | 1584x396 | 4:1, 1K | none | 1584x396 |
| LinkedIn company banner | 1128x191 | 8:1, 1K | 402, 0, 2268, 384 | 1128x191 |
| Google Play feature graphic | 1024x500 | 21:9, 1K | 104, 0, 1376, 672 | 1024x500 |
| dev.to cover | 1000x420 | 21:9, 1K | none | 1000x420 |
| Instagram square | 1080x1080 | 1:1, 2K | none | 1080x1080 |
| Instagram portrait | 1080x1350 | 4:5, 2K | none | 1080x1350 |
| Instagram story, Reels, TikTok | 1080x1920 | 9:16, 2K | 0, 10, 1536, 2731 | 1080x1920 |
| README hero | about 1376x768 | 16:9, 1K | none | none |
| Landing hero, full bleed | up to 1920 wide | 21:9, 2K | none | 1920x1080 with `maintain_aspect=true` (yields 1920x815) |
| Square app icon master | 1024x1024 | 1:1, 1K | none | 512, 192, 180, 48, 32, 16 with `maintain_aspect=true` |

## 4. Icon file set, tags and manifest

Every file is a `transform_image(operation="resize")` of the 1024 master; never
derive a size from a smaller output.

| File | Pixels | Used by | Note |
|---|---|---|---|
| favicon-16.png | 16 | browser tab | |
| favicon-32.png | 32 | tab, bookmarks, taskbar | |
| favicon-48.png | 48 | Windows site icon, ICO member | |
| favicon.ico | 16, 32, 48 | legacy browsers and bare `/favicon.ico` requests | packed outside the server (Pillow or ImageMagick) |
| apple-touch-icon.png | 180 | iOS home screen | full-bleed square; iOS applies its own rounded mask |
| icon-192.png | 192 | Android home screen, PWA install | manifest |
| icon-512.png | 512 | PWA splash, store listings | manifest; `maskable` only if the glyph sits in the central 80 percent |

```html
<link rel="icon" href="/favicon.ico" sizes="any">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16.png">
<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">
<link rel="manifest" href="/site.webmanifest">
```

```json
{
  "icons": [
    { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable" }
  ]
}
```

Next.js App Router emits the tags itself when `icon.png`, `apple-icon.png` and
`opengraph-image.png` sit in `app/`; copy the finished files there under those names.

## 5. Safe zones

| Asset | Keep important content inside |
|---|---|
| Open Graph 1200x630 | 60 px from every edge; each platform crops differently |
| X header 1500x500 | the central 1500x360 band; the profile photo covers the bottom left |
| YouTube banner 2560x1440 | the central 1546x423 area, which every device shows |
| Maskable PWA icon 512x512 | a centred circle 410 px across (80 percent); corners can be cut |
| apple-touch-icon 180x180 | full bleed; iOS rounds the corners itself |
| Landing hero behind a headline | leave one third of the width empty on the headline side |
