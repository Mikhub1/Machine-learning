# generate_mock_ui.py
# Produces a PNG mockup of the "expected final product" dashboard using Pillow.
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 800
bg = (245, 246, 248)  # light-ish background
card = (255, 255, 255)
text = (20, 20, 20)
muted = (90, 90, 90)
accent = (66, 133, 244)

img = Image.new('RGB', (W, H), bg)
draw = ImageDraw.Draw(img)

# Card
pad = 40
draw.rounded_rectangle((pad, pad, W-pad, H-pad), radius=28, fill=card)

# Title
try:
    font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 44)
    font_h = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    font_p = ImageFont.truetype("DejaVuSans.ttf", 26)
except:
    font_title = ImageFont.load_default()
    font_h = ImageFont.load_default()
    font_p = ImageFont.load_default()

draw.text((pad+30, pad+24), "Job Application Assistance Model", fill=text, font=font_title)

# Score badge
badge_x, badge_y = pad+30, pad+100
draw.rounded_rectangle((badge_x, badge_y, badge_x+120, badge_y+46), 12, fill=accent)
draw.text((badge_x+18, badge_y+10), "Score", fill=(255,255,255), font=font_h)

# Big score
draw.text((badge_x, badge_y+70), "0.87", fill=text, font=font_title)

# Columns
col1_x = pad+30
col2_x = W//2
y0 = badge_y + 150

def tag(draw, xy, text_str):
    x, y = xy
    tw, th = draw.textsize(text_str, font=font_p)
    draw.rounded_rectangle((x, y, x+tw+22, y+th+14), 12, fill=(238, 240, 244))
    draw.text((x+11, y+7), text_str, fill=(0,0,0), font=font_p)
    return x+tw+22+10, y

# Headings
draw.text((col1_x, y0), "Resume Skills", fill=muted, font=font_h)
draw.text((col2_x, y0), "Job Skills", fill=muted, font=font_h)

# Resume skill chips
skills_resume = ["Python", "SQL", "Pandas", "Git", "Docker"]
x, y = col1_x, y0+40
for s in skills_resume:
    x, y = tag(draw, (x, y), s)
draw.text((col1_x, y+60), "Years in Resume: 2", fill=muted, font=font_p)

# Job skill chips
skills_job = ["Python", "SQL", "Tableau", "Power BI", "Pandas", "Git"]
x, y = col2_x, y0+40
for s in skills_job:
    x, y = tag(draw, (x, y), s)
draw.text((col2_x, y+60), "Years in Job: 3", fill=muted, font=font_p)

# Missing skills
ms_y = y0+220
draw.text((col1_x, ms_y), "Missing Skills", fill=muted, font=font_h)
x, y = col1_x, ms_y+42
for s in ["Tableau", "Power BI"]:
    x, y = tag(draw, (x, y), s)

# Save
out_path = "/mnt/data/mock_dashboard.png"
img.save(out_path)
print("Saved:", out_path)
