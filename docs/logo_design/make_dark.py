"""Generate a dark-mode variant of logo3 by fading bright desaturated pixels."""

import numpy as np
from PIL import Image

src = Image.open("logo3.png").convert("RGBA")
arr = np.array(src, dtype=np.float32)
rgb, a = arr[..., :3], arr[..., 3]

# luminance (Rec. 709) and saturation (max-min / max)
lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
mx, mn = rgb.max(-1), rgb.min(-1)
sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)

# fade factor: bright + desaturated → fade toward 0; colored stays
bright = np.clip((lum - 150) / 90, 0, 1)  # ramps 150..240 → 0..1
desat = np.clip(1 - sat * 4, 0, 1)  # sat < 0.25 counts as "neutral"
fade = bright * desat  # 1 = fully fade (halo), 0 = keep

# shrink alpha, and pull RGB toward black for the faded parts
new_a = a * (1 - 0.9 * fade)
new_rgb = rgb * (1 - 0.55 * fade[..., None])

out = np.concatenate([new_rgb, new_a[..., None]], axis=-1)
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save("logo3_dark.png")
print("wrote logo3_dark.png")
