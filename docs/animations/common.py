# docs/animations/common.py
from manimlib import *

# ── Color palette (light/diagram theme) ──────────────────────
TEXT_COLOR = "#333333"
ACCENT_BLUE = "#4A90D9"
ACCENT_GREEN = "#5CB85C"
ACCENT_ORANGE = "#F0AD4E"
ACCENT_RED = "#D9534F"
INACTIVE_GRAY = "#CCCCCC"
BORDER_COLOR = "#888888"
BOX_FILL = "#FFFFFF"

def labeled_box(label, color=BORDER_COLOR, width=2.0, height=0.7, font_size=22,
                fill=None, text_color=None):
    box = RoundedRectangle(
        width=width, height=height, corner_radius=0.12,
        stroke_color=color, stroke_width=2,
        fill_color=fill or BOX_FILL, fill_opacity=0.95,
    )
    txt = Text(str(label), font_size=font_size)
    txt.set_color(text_color or TEXT_COLOR)
    txt.set_max_width(width - 0.2)
    txt.move_to(box.get_center())
    return VGroup(box, txt)

PROMPT_FILL = "#DDEAF6"
PROMPT_COLOR = "#3A7BBF"
LLM_FILL = "#E8E8E8"
LLM_COLOR = "#555555"

def thin_arrow(start, end, color=BORDER_COLOR):
    return Arrow(
        start, end, buff=0.1,
        fill_color=color, stroke_width=0,
        thickness=2,
        max_tip_length_to_length_ratio=0.15,
    )
