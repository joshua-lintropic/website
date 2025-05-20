from PIL import Image

# 1. Open source and compute new size
src = Image.open("logo-1-color.png")
w_src, h_src = src.size
new_w = 209
new_h = int(h_src * (new_w / w_src))
resized = src.resize((new_w, new_h), Image.LANCZOS)

# 2. Create blank canvas
canvas_h = 131
canvas = Image.new("RGB", (new_w, canvas_h), (255, 255, 255))  # white bg

# 3. Compute vertical offset to center
y_offset = (canvas_h - new_h) // 2
canvas.paste(resized, (0, y_offset))

# 4. Save result
canvas.save("logo-1-color-resized.png")
