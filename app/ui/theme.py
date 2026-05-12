"""FaceDoodle 设计系统 tokens。

基于 DESIGN.md。所有交互元素使用单一主色调。无装饰渐变，无 chrome 阴影。
"""

# ── 色彩 ──────────────────────────────────────────────────────────────────

# 品牌色 & 强调色
PRIMARY = "#0066cc"            # 主色调 — 唯一的交互色
PRIMARY_FOCUS = "#0071e3"      # 聚焦环色
PRIMARY_ON_DARK = "#2997ff"    # 暗色表面上的链接色

# 表面色
CANVAS = "#ffffff"             # 纯白 — 主内容区
PARCHMENT = "#f5f5f7"          # 米白 — 侧边栏、底部、备选卡片
SURFACE_PEARL = "#fafafc"      # 近白 — 幽灵按钮填充
SURFACE_TILE_1 = "#272729"     # 近黑 — 暗色卡片主色
SURFACE_TILE_2 = "#2a2a2c"     # 近黑微亮
SURFACE_TILE_3 = "#252527"     # 近黑微暗
SURFACE_BLACK = "#000000"      # 纯黑 — 导航栏背景
CHIP_TRANSLUCENT = "#d2d2d7"   # 半透明灰色芯片底色

# 文字色
INK = "#1d1d1f"                # 近黑 — 亮色表面上所有文字
INK_MUTED_80 = "#333333"       # 柔和正文
INK_MUTED_48 = "#7a7a7a"       # 禁用态 / 法律声明文字
BODY_ON_DARK = "#ffffff"       # 暗色表面上的文字
BODY_MUTED = "#cccccc"         # 暗色表面上的次要文字

# 分割线 & 边框
DIVIDER_SOFT = "#f0f0f0"       # 幽灵按钮的柔和环
HAIRLINE = "#e0e0e0"           # 1px 卡片边框
HAIRLINE_TINT = "rgba(0, 0, 0, 0.08)"  # 半透明分割线

# 破坏性操作色（红色，保留给删除等操作）
DESTRUCTIVE = "#e34b4b"
DESTRUCTIVE_HOVER = "#f06060"
ERROR = "#e53e3e"              # 错误消息文字

# ── 字体排版 ─────────────────────────────────────────────────────────────
# QSS 近似的 SF Pro 层级。
# Windows 上 system-ui 解析为 Segoe UI Variable；中文使用 PingFang SC。
# Qt 不支持 letter-spacing 和精确 line-height，通过 padding 近似 line-height，
# 忽略负 letter-spacing。

FONT_STACK = ('"Segoe UI Variable", "PingFang SC", "Microsoft YaHei", '
              'system-ui, -apple-system, sans-serif')

# Font tokens as (size_px, weight_str) tuples for building QSS
TYPO = {
    "hero":       ("56px", "600"),
    "display-lg": ("40px", "600"),
    "display-md": ("34px", "600"),
    "lead":       ("28px", "400"),
    "lead-airy":  ("24px", "300"),
    "tagline":    ("21px", "600"),
    "body-strong":("17px", "600"),
    "body":       ("17px", "400"),
    "caption-strong": ("14px", "600"),
    "caption":    ("14px", "400"),
    "button-large": ("18px", "300"),
    "button":     ("17px", "400"),
    "button-utility": ("14px", "400"),
    "fine-print": ("12px", "400"),
    "micro-legal":("10px", "400"),
    "nav-link":   ("12px", "400"),
}


# Precomputed font CSS fragments (no per-call dict lookup or f-string construction)
_FONT_CSS = {}
for _token, (_size, _weight) in TYPO.items():
    _FONT_CSS[_token] = f"font-family: {FONT_STACK}; font-size: {_size}; font-weight: {_weight};"


def font_css(token, color=None):
    """返回预计算的 QSS 字体片段，可选附带颜色。"""
    css = _FONT_CSS.get(token, _FONT_CSS["body"])
    if color:
        return f"{css} color: {color};"
    return css


def label_css(token, color=None):
    """返回 QLabel 样式：字体 + 透明背景 + 无边框。"""
    css = _FONT_CSS.get(token, _FONT_CSS["body"])
    if color:
        return f"{css} color: {color}; background: transparent; border: none;"
    return f"{css} background: transparent; border: none;"


# ── Spacing ──────────────────────────────────────────────────────────────

SPACE = {
    "xxs": 4,
    "xs": 8,
    "sm": 12,
    "md": 17,
    "lg": 24,
    "xl": 32,
    "xxl": 48,
    "section": 80,
}

# ── Rounded Corners ──────────────────────────────────────────────────────

ROUNDED = {
    "none": "0px",
    "xs": "5px",
    "sm": "8px",
    "md": "11px",
    "lg": "18px",
    "pill": "9999px",
    "full": "9999px",
}


# ── Helpers ──────────────────────────────────────────────────────────────

_RGBA_CACHE = {}


def rgba(hex_color, alpha):
    """将 hex 颜色转为 rgba() 字符串。按 (hex, alpha) 缓存。"""
    a_rounded = round(alpha, 2)
    key = (hex_color, a_rounded)
    cached = _RGBA_CACHE.get(key)
    if cached is not None:
        return cached
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    result = f"rgba({r}, {g}, {b}, {alpha})"
    _RGBA_CACHE[key] = result
    return result


def _no_border():
    return "border: none;"


# ── Global Stylesheet ────────────────────────────────────────────────────

# Precomputed at module load to avoid recomputation on every call
_GLOBAL_STYLESHEET = f"""
    QMainWindow {{
        background: {CANVAS};
    }}
    QWidget {{
        font-family: {FONT_STACK};
        color: {INK};
    }}
    QLabel {{
        color: {INK};
        background: transparent;
        border: none;
    }}
    QLineEdit {{
        background: {CANVAS};
        color: {INK};
        border: 1px solid {HAIRLINE};
        border-radius: {ROUNDED["pill"]};
        padding: 12px 18px;
        {font_css("body")}
    }}
    QLineEdit:focus {{
        border-color: {PRIMARY};
    }}
    QLineEdit:disabled {{
        background: {DIVIDER_SOFT};
        color: {INK_MUTED_48};
    }}
    QMessageBox {{
        background: {CANVAS};
    }}
    QMessageBox QLabel {{
        color: {INK};
        {font_css("body")}
    }}
    QMessageBox QPushButton {{
        background: {PRIMARY};
        color: {CANVAS};
        border: none;
        border-radius: {ROUNDED["pill"]};
        padding: 8px 24px;
        {font_css("caption")}
    }}
    QMessageBox QPushButton:hover {{
        background: {PRIMARY_FOCUS};
    }}
    QComboBox {{
        background: {CANVAS};
        color: {INK};
        border: 1px solid {HAIRLINE};
        border-radius: {ROUNDED["sm"]};
        padding: 4px 8px;
        {font_css("caption")}
    }}
    QComboBox:hover {{
        border-color: {PRIMARY};
    }}
    QComboBox QAbstractItemView {{
        background: {CANVAS};
        color: {INK};
        selection-background-color: {PRIMARY};
        selection-color: {CANVAS};
        border: 1px solid {HAIRLINE};
    }}
    QCheckBox {{
        color: {INK};
        {font_css("caption")}
        spacing: 6px;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {HAIRLINE};
        border-radius: {ROUNDED["xs"]};
        background: {CANVAS};
    }}
    QCheckBox::indicator:checked {{
        background: {PRIMARY};
        border-color: {PRIMARY};
    }}
    QSlider::groove:horizontal {{
        background: {HAIRLINE};
        height: 4px;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {PRIMARY};
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
    }}
    QScrollBar::handle:vertical {{
        background: {rgba(INK, 0.2)};
        min-height: 30px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {rgba(INK, 0.35)};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 8px;
    }}
    QScrollBar::handle:horizontal {{
        background: {rgba(INK, 0.2)};
        min-width: 30px;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {rgba(INK, 0.35)};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}
"""


def global_stylesheet():
    """返回应用到主窗口的全局 QSS。"""
    return _GLOBAL_STYLESHEET


# ── Button Styles ────────────────────────────────────────────────────────

def pill_button_style(color=PRIMARY, text_color=CANVAS):
    """主要 pill CTA 按钮样式。"""
    return f"""
        QPushButton {{
            background: {color};
            color: {text_color};
            border: none;
            border-radius: {ROUNDED["pill"]};
            padding: 11px 22px;
            {font_css("body")}
        }}
        QPushButton:hover {{
            background: {PRIMARY_FOCUS};
        }}
        QPushButton:pressed {{
            background: {color};
        }}
        QPushButton:disabled {{
            background: {HAIRLINE};
            color: {INK_MUTED_48};
        }}
    """


def ghost_pill_button_style(text_color=PRIMARY):
    """Secondary pill — transparent bg, colored text, 1px colored border."""
    return f"""
        QPushButton {{
            background: transparent;
            color: {text_color};
            border: 1px solid {text_color};
            border-radius: {ROUNDED["pill"]};
            padding: 10px 21px;
            {font_css("body")}
        }}
        QPushButton:hover {{
            background: {rgba(PRIMARY, 0.08)};
        }}
        QPushButton:pressed {{
            background: {rgba(PRIMARY, 0.15)};
        }}
        QPushButton:disabled {{
            color: {INK_MUTED_48};
            border-color: {HAIRLINE};
            background: transparent;
        }}
    """


def ghost_destructive_button_style():
    """Ghost pill for destructive actions (delete)."""
    return f"""
        QPushButton {{
            background: transparent;
            color: {DESTRUCTIVE};
            border: 1px solid {DESTRUCTIVE};
            border-radius: {ROUNDED["pill"]};
            padding: 10px 21px;
            {font_css("body")}
        }}
        QPushButton:hover {{
            background: {rgba(DESTRUCTIVE, 0.08)};
        }}
        QPushButton:pressed {{
            background: {rgba(DESTRUCTIVE, 0.15)};
        }}
        QPushButton:disabled {{
            color: {INK_MUTED_48};
            border-color: {HAIRLINE};
            background: transparent;
        }}
    """


def utility_button_style():
    """Compact utility button — small radius, subtle."""
    return f"""
        QPushButton {{
            background: {CANVAS};
            color: {INK};
            border: 1px solid {HAIRLINE};
            border-radius: {ROUNDED["sm"]};
            padding: 6px 14px;
            {font_css("caption")}
        }}
        QPushButton:hover {{
            background: {DIVIDER_SOFT};
        }}
        QPushButton:pressed {{
            background: {HAIRLINE};
        }}
        QPushButton:checked {{
            background: {PRIMARY};
            color: {CANVAS};
            border-color: {PRIMARY};
        }}
    """


def utility_danger_button_style():
    """Utility button that turns red on hover (clear button)."""
    return f"""
        QPushButton {{
            background: {CANVAS};
            color: {INK};
            border: 1px solid {HAIRLINE};
            border-radius: {ROUNDED["sm"]};
            padding: 6px 14px;
            {font_css("caption")}
        }}
        QPushButton:hover {{
            background: {rgba(DESTRUCTIVE, 0.08)};
            color: {DESTRUCTIVE};
            border-color: {DESTRUCTIVE};
        }}
    """


def checkable_pill_button_style():
    """Toggle button — ghost when off, filled Action Blue when checked."""
    return f"""
        QPushButton {{
            background: transparent;
            color: {PRIMARY};
            border: 1px solid {PRIMARY};
            border-radius: {ROUNDED["pill"]};
            padding: 10px 21px;
            {font_css("body")}
        }}
        QPushButton:hover {{
            background: {rgba(PRIMARY, 0.08)};
        }}
        QPushButton:checked {{
            background: {PRIMARY};
            color: {CANVAS};
        }}
        QPushButton:checked:hover {{
            background: {PRIMARY_FOCUS};
        }}
        QPushButton:disabled {{
            color: {INK_MUTED_48};
            border-color: {HAIRLINE};
            background: transparent;
        }}
    """


# ── Panel / Card Styles ──────────────────────────────────────────────────

def parchment_panel_style():
    """Side panel with parchment background."""
    return f"""
        background: {PARCHMENT};
        border: none;
    """


def card_style(selected=False):
    """Thumbnail card border."""
    border_color = PRIMARY if selected else HAIRLINE
    return f"""
        background: {DIVIDER_SOFT};
        border: 2px solid {border_color};
        border-radius: {ROUNDED["xs"]};
    """


def frosted_bar_style():
    """Top bar / sub-nav — parchment with hairline bottom border."""
    return f"""
        background: {PARCHMENT};
        border-bottom: 1px solid {HAIRLINE};
    """
