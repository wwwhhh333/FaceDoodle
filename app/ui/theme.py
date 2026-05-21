"""FaceDoodle design system — Bugatti / inspired by DESIGN.md.

Pure black canvas, white uppercase-spaced display, weight-400 throughout,
transparent pill buttons, rect cards, no accent — adapted for Chinese reading.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Color palette — bugatti monochrome (DESIGN.md §2)
# ══════════════════════════════════════════════════════════════════════════════

CANVAS         = "#ffffff"    # white — page floor
SURFACE_SOFT   = "#f5f5f5"   # light gray — panels
SURFACE_CARD   = "#ffffff"   # card surface
SURFACE_NAV    = "#00796b"   # dark teal — top bar
SURFACE_HOVER  = "#e0f2f1"   # very light teal — hover
SURFACE_ELEVATED = "#fafafa" # elevated surface

INK            = "#212121"   # primary text
BODY           = "#757575"   # secondary text
BODY_STRONG    = "#212121"   # emphasized body
MUTED          = "#757575"   # captions
MUTED_SOFT     = "#9e9e9e"  # fine print

HAIRLINE       = "#BDBDBD"   # divider
HAIRLINE_STRONG = "#9e9e9e"  # heavier divider

ACCENT         = "#cddc39"   # yellow — accent
ACCENT_HOVER   = "#fdd835"   # darker yellow
ACCENT_MUTED   = "rgba(255, 235, 59, 0.15)"  # subtle yellow wash

PRIMARY        = "#009688"   # teal — main brand
PRIMARY_HOVER  = "#00796b"   # dark teal
PRIMARY_MUTED  = "rgba(0, 150, 136, 0.12)"  # subtle teal wash
LIGHT_TEAL     = "#b2dfdb"   # light teal

LINK           = "#009688"   # teal for links

WARNING        = "#d4a017"   # warning states
SUCCESS        = "#5fa657"   # success states (rarely appears)
ERROR_CRIMSON  = "#b53333"   # warm red — error states

# ── Legacy aliases ──
TERRACOTTA     = PRIMARY
CORAL          = ACCENT
PRIMARY_FOCUS  = PRIMARY_HOVER
PRIMARY_ON_DARK = LIGHT_TEAL
DESTRUCTIVE    = ERROR_CRIMSON
ERROR          = ERROR_CRIMSON
PARCHMENT      = SURFACE_CARD
IVORY          = SURFACE_CARD
PURE_WHITE     = INK
WARM_SAND      = SURFACE_CARD
DARK_SURFACE   = SURFACE_CARD
NEAR_BLACK     = INK
CHARCOAL_WARM  = BODY
OLIVE_GRAY     = MUTED
STONE_GRAY     = MUTED_SOFT
WARM_SILVER    = BODY_STRONG
SURFACE_BLACK  = CANVAS
SURFACE_TILE_1 = SURFACE_CARD
CHIP_TRANSLUCENT = SURFACE_CARD
BORDER_CREAM   = HAIRLINE
BORDER_WARM    = HAIRLINE_STRONG
RING_WARM      = HAIRLINE_STRONG
RING_DEEP      = HAIRLINE_STRONG
DIVIDER_SOFT   = HAIRLINE
BORDER_DARK    = HAIRLINE
FOCUS_BLUE     = INK
INK_MUTED_48   = MUTED_SOFT
INK_MUTED_80   = MUTED
HAIRLINE_TINT  = HAIRLINE
SURFACE_PEARL  = SURFACE_CARD
BODY_ON_DARK   = BODY

# ══════════════════════════════════════════════════════════════════════════════
# Typography — Bugatti trinity adapted for Chinese
#
# Bugatti uses: Display (uppercase, wide-tracking), Text Regular (serif body),
# Monospace (buttons/captions/nav).  All at weight 400 — no bold anywhere.
#
# Chinese adaptation:
# - Headlines use Source Han Sans SC (matching Bugatti Display's sans) at 400
# - Body uses Source Han Serif SC (matching Bugatti Text Regular's serif)
# - Line-height ≥ 1.60 for Chinese readability
# - CJK doesn't use letter-spacing — negative tracking would compress kerning
# ══════════════════════════════════════════════════════════════════════════════

FONT_DISPLAY = ('"Source Han Sans SC", "Noto Sans SC", "PingFang SC", '
                '"Segoe UI", -apple-system, sans-serif')
FONT_SERIF   = ('"Source Han Serif SC", "Noto Serif SC", '
                'Georgia, "SimSun", serif')
FONT_MONO    = ('"JetBrains Mono", "Cascadia Mono", "IBM Plex Mono", '
                'ui-monospace, "SF Mono", monospace')

# (size, weight, line_height)
TYPO = {
    "hero":         ("48px", "600", 1.25),
    "display":      ("36px", "600", 1.30),
    "heading":      ("28px", "600", 1.35),
    "title":        ("22px", "600", 1.35),
    "lead":         ("18px", "500", 1.75),
    "body":         ("16px", "400", 1.75),
    "body-small":   ("15px", "400", 1.70),
    "body-strong":  ("16px", "600", 1.70),
    "caption":      ("14px", "500", 1.60),
    "caption-strong": ("14px", "600", 1.55),
    "label":        ("13px", "500", 1.50),
    "fine-print":   ("12px", "400", 1.50),
    "button":       ("15px", "600", 1.0),
    "button-utility": ("13px", "500", 1.0),
    "button-large": ("16px", "500", 1.0),
    "nav-link":     ("13px", "500", 1.4),
    "tagline":      ("16px", "600", 1.0),
}

_FONT_CSS = {}
for _token, (_size, _weight, _lh) in TYPO.items():
    _FONT_CSS[_token] = (f"font-family: {FONT_SERIF}; "
                         f"font-size: {_size}; font-weight: {_weight};")


def font_css(token, color=None):
    css = _FONT_CSS.get(token, _FONT_CSS["body"])
    if color:
        return f"{css} color: {color};"
    return css


def serif_css(token, color=None):
    _size, _weight, _lh = TYPO.get(token, TYPO["heading"])
    css = (f"font-family: {FONT_SERIF}; "
           f"font-size: {_size}; font-weight: {_weight};")
    if color:
        return f"{css} color: {color};"
    return css


def label_css(token, color=None):
    css = _FONT_CSS.get(token, _FONT_CSS["body"])
    if color:
        return f"{css} color: {color}; background: transparent; border: none;"
    return f"{css} background: transparent; border: none;"


# ══════════════════════════════════════════════════════════════════════════════
# Spacing — 4px base (DESIGN.md §5)
# ══════════════════════════════════════════════════════════════════════════════

SPACE = {
    "xxs": 4, "xs": 8, "sm": 12, "md": 16, "lg": 24, "xl": 40, "xxl": 64,
    "section": 80,
}

# ══════════════════════════════════════════════════════════════════════════════
# Shapes — binary radius (DESIGN.md §5): 0px for everything, pill for buttons
# ══════════════════════════════════════════════════════════════════════════════

ROUNDED = {
    "none":  "0px",
    "xs":    "6px",
    "sm":    "8px",
    "md":    "12px",
    "lg":    "16px",
    "xl":    "20px",
    "full":  "9999px",
    "pill":  "9999px",
}

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

_RGBA_CACHE = {}


def rgba(hex_color, alpha):
    a_rounded = round(alpha, 2)
    key = (hex_color, a_rounded)
    cached = _RGBA_CACHE.get(key)
    if cached:
        return cached
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    result = f"rgba({r}, {g}, {b}, {alpha})"
    _RGBA_CACHE[key] = result
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Global stylesheet — Bugatti monochrome
# ══════════════════════════════════════════════════════════════════════════════

_GLOBAL_STYLESHEET = f"""
    QMainWindow {{
        background: {CANVAS};
    }}
    QWidget {{
        font-family: {FONT_SERIF};
        color: {INK};
        selection-background-color: {ACCENT_MUTED};
        selection-color: {INK};
    }}
    QLabel {{
        color: {BODY};
        background: transparent;
        border: none;
    }}

    QToolTip {{
        background: {SURFACE_CARD};
        color: {BODY};
        border: 1px solid {HAIRLINE};
        border-radius: {ROUNDED["none"]};
        padding: 6px 10px;
        {font_css("fine-print")}
    }}

    QMenu {{
        background: {SURFACE_CARD};
        border: 1px solid {HAIRLINE};
        border-radius: {ROUNDED["none"]};
        padding: 4px 0;
    }}
    QMenu::item {{
        padding: 8px 24px 8px 16px;
        {font_css("label")}
    }}
    QMenu::item:selected {{
        background: {SURFACE_ELEVATED};
        color: {INK};
    }}
    QMenu::separator {{
        height: 1px;
        background: {HAIRLINE};
        margin: 4px 8px;
    }}

    /* ── Input fields — transparent, bottom border only ── */
    QLineEdit {{
        background: {SURFACE_SOFT};
        color: {INK};
        border: 1px solid {HAIRLINE};
        border-radius: {ROUNDED["none"]};
        padding: 10px 16px;
        {font_css("body")}
    }}
    QLineEdit:focus {{
        border-color: {ACCENT};
        border-width: 1.5px;
    }}
    QLineEdit:disabled {{
        background: {SURFACE_CARD};
        color: {MUTED};
    }}

    /* ── Spin boxes ── */
    QSpinBox, QDoubleSpinBox {{
        background: {SURFACE_SOFT};
        color: {INK};
        border: 1px solid {HAIRLINE};
        border-radius: {ROUNDED["none"]};
        padding: 6px 10px;
        {font_css("label")}
    }}
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {ACCENT};
        border-width: 1.5px;
    }}
    QSpinBox:disabled, QDoubleSpinBox:disabled {{
        background: {SURFACE_CARD};
        color: {MUTED};
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        border: none;
        border-left: 1px solid {HAIRLINE};
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        border: none;
        border-left: 1px solid {HAIRLINE};
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-bottom: 5px solid {MUTED};
        width: 0; height: 0;
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {MUTED};
        width: 0; height: 0;
    }}

    /* ── ComboBox ── */
    QComboBox {{
        background: {SURFACE_SOFT};
        color: {INK};
        border: 1px solid {HAIRLINE};
        border-radius: {ROUNDED["none"]};
        padding: 6px 12px;
        {font_css("label")}
    }}
    QComboBox:hover {{ border-color: {HAIRLINE_STRONG}; }}
    QComboBox:on {{ border-color: {ACCENT}; }}
    QComboBox:disabled {{
        background: {SURFACE_CARD};
        color: {MUTED};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {MUTED};
        width: 0; height: 0;
        margin-right: 4px;
    }}
    QComboBox QAbstractItemView {{
        background: {SURFACE_CARD};
        color: {INK};
        selection-background-color: {SURFACE_ELEVATED};
        selection-color: {INK};
        border: 1px solid {HAIRLINE};
        border-radius: {ROUNDED["none"]};
        padding: 2px 0;
        outline: none;
    }}

    /* ── Checkbox ── */
    QCheckBox {{
        color: {BODY};
        {font_css("label")}
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 18px; height: 18px;
        border: 1.5px solid {HAIRLINE_STRONG};
        border-radius: {ROUNDED["none"]};
        background: {SURFACE_SOFT};
    }}
    QCheckBox::indicator:hover {{
        border-color: {ACCENT};
    }}
    QCheckBox::indicator:checked {{
        background: {ACCENT};
        border-color: {ACCENT};
    }}
    QCheckBox:disabled {{ color: {MUTED}; }}
    QCheckBox::indicator:disabled {{
        background: {SURFACE_CARD};
        border-color: {HAIRLINE};
    }}

    /* ── Sliders ── */
    QSlider::groove:horizontal {{
        background: {HAIRLINE};
        height: 4px;
        border-radius: {ROUNDED["none"]};
    }}
    QSlider::handle:horizontal {{
        background: {ACCENT};
        width: 16px; height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {ACCENT_HOVER};
    }}

    /* ── Progress bar ── */
    QProgressBar {{
        background: {HAIRLINE};
        border: none;
        border-radius: {ROUNDED["none"]};
        height: 4px;
        text-align: center;
        color: {MUTED};
        {font_css("fine-print")}
    }}
    QProgressBar::chunk {{
        background: {ACCENT};
        border-radius: {ROUNDED["xs"]};
    }}

    /* ── Scroll bars — minimal ── */
    QScrollBar:vertical {{
        background: transparent;
        width: 4px; margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {rgba(BODY, 0.18)};
        min-height: 36px;
        border-radius: {ROUNDED["none"]};
    }}
    QScrollBar::handle:vertical:hover {{
        background: {rgba(ACCENT, 0.50)};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {rgba(BODY, 0.18)};
        min-width: 36px;
        border-radius: {ROUNDED["none"]};
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {rgba(ACCENT, 0.50)};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

    QScrollArea {{ background: transparent; border: none; }}

    /* ── Splitter handle — 1px hairline ── */
    QSplitter::handle {{
        background: {rgba(BODY, 0.08)};
        border-radius: {ROUNDED["none"]};
    }}
    QSplitter::handle:hover {{
        background: {rgba(ACCENT, 0.35)};
    }}
    QSplitter::handle:horizontal {{ width: 1px; }}
    QSplitter::handle:vertical   {{ height: 1px; }}

    /* ── Group box ── */
    QGroupBox {{
        {font_css("label")}
        border: 1px solid {HAIRLINE};
        border-radius: {ROUNDED["none"]};
        margin-top: 16px;
        padding: 16px 12px 12px 12px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 6px;
        color: {BODY};
    }}

    /* ── List widget ── */
    QListWidget {{
        background: {SURFACE_SOFT};
        border: 1px solid {HAIRLINE};
        border-radius: {ROUNDED["none"]};
        padding: 4px;
        outline: none;
        {font_css("label")}
    }}
    QListWidget::item {{
        padding: 8px 12px;
        border-radius: {ROUNDED["none"]};
    }}
    QListWidget::item:selected {{
        background: {ACCENT_MUTED};
        color: {ACCENT};
    }}
    QListWidget::item:hover {{
        background: {SURFACE_HOVER};
    }}

    /* ── Dialogs ── */
    QDialog {{
        background: {SURFACE_SOFT};
        border: 1px solid {HAIRLINE};
        border-radius: {ROUNDED["none"]};
    }}
    QMessageBox {{
        background: {SURFACE_CARD};
        border-radius: {ROUNDED["none"]};
    }}
    QMessageBox QLabel {{
        color: {BODY};
        {font_css("body")}
    }}
    QMessageBox QPushButton {{
        background: {ACCENT};
        color: #ffffff;
        border: none;
        border-radius: {ROUNDED["pill"]};
        padding: 8px 24px;
        {font_css("button")}
    }}
    QMessageBox QPushButton:hover {{
        background: {ACCENT_HOVER};
    }}

    /* ── Text edit ── */
    QTextEdit, QPlainTextEdit {{
        background: {SURFACE_SOFT};
        color: {INK};
        border: 1px solid {HAIRLINE};
        border-radius: {ROUNDED["none"]};
        padding: 8px 12px;
        {font_css("body")}
        selection-background-color: {ACCENT_MUTED};
        selection-color: {INK};
    }}
    QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {INK};
        border-width: 1.5px;
    }}
"""


def global_stylesheet():
    return _GLOBAL_STYLESHEET


# ══════════════════════════════════════════════════════════════════════════════
# Button styles — transparent + outline only (DESIGN.md §4)
# ══════════════════════════════════════════════════════════════════════════════

_BTN_BASE = "QPushButton"


def transport_button_style():
    return f"""
        {_BTN_BASE} {{
            background: transparent;
            color: {BODY};
            border: 1px solid {HAIRLINE};
            border-radius: {ROUNDED["sm"]};
            padding: 5px 11px;
            min-width: 28px;
            {font_css("button-utility")}
        }}
        {_BTN_BASE}:hover {{
            background: {PRIMARY_MUTED};
            border-color: {PRIMARY};
        }}
        {_BTN_BASE}:pressed {{
            background: {rgba(PRIMARY, 0.22)};
        }}
        {_BTN_BASE}:checked {{
            background: {PRIMARY};
            color: #ffffff;
            border-color: {PRIMARY};
        }}
        {_BTN_BASE}:disabled {{
            color: {MUTED};
            border-color: {HAIRLINE};
        }}
    """


def pill_button_style(color=PRIMARY, text_color="#ffffff"):
    """Primary CTA — teal filled, or custom color."""
    return f"""
        {_BTN_BASE} {{
            background: {color};
            color: {text_color};
            border: none;
            border-radius: {ROUNDED["sm"]};
            padding: 11px 28px;
            {font_css("button")}
        }}
        {_BTN_BASE}:hover {{ background: {PRIMARY_HOVER}; }}
        {_BTN_BASE}:pressed {{ background: {color}; }}
        {_BTN_BASE}:disabled {{
            background: {rgba(BODY, 0.20)};
            color: {MUTED};
        }}
    """


def ghost_pill_button_style(text_color=PRIMARY):
    return f"""
        {_BTN_BASE} {{
            background: transparent;
            color: {text_color};
            border: 1px solid {HAIRLINE};
            border-radius: {ROUNDED["sm"]};
            padding: 11px 28px;
            {font_css("button")}
        }}
        {_BTN_BASE}:hover {{
            background: {PRIMARY_MUTED};
            border-color: {PRIMARY};
        }}
        {_BTN_BASE}:pressed {{
            background: {rgba(PRIMARY, 0.22)};
        }}
        {_BTN_BASE}:disabled {{
            color: {MUTED};
            border-color: {HAIRLINE};
        }}
    """


def ghost_destructive_button_style():
    return f"""
        {_BTN_BASE} {{
            background: transparent;
            color: {ERROR_CRIMSON};
            border: 1px solid {HAIRLINE};
            border-radius: {ROUNDED["sm"]};
            padding: 10px 28px;
            {font_css("button")}
        }}
        {_BTN_BASE}:hover {{
            background: {rgba(ERROR_CRIMSON, 0.10)};
            border-color: {ERROR_CRIMSON};
        }}
        {_BTN_BASE}:pressed {{
            background: {rgba(ERROR_CRIMSON, 0.22)};
        }}
        {_BTN_BASE}:disabled {{
            color: {MUTED};
            border-color: {HAIRLINE};
        }}
    """


def utility_button_style():
    return f"""
        {_BTN_BASE} {{
            background: transparent;
            color: {BODY};
            border: 1px solid {HAIRLINE};
            border-radius: {ROUNDED["sm"]};
            padding: 6px 14px;
            {font_css("label")}
        }}
        {_BTN_BASE}:hover {{
            background: {PRIMARY_MUTED};
            border-color: {PRIMARY};
            color: {PRIMARY};
        }}
        {_BTN_BASE}:pressed {{
            background: {rgba(PRIMARY, 0.22)};
        }}
        {_BTN_BASE}:checked {{
            background: {PRIMARY};
            color: #ffffff;
            border-color: {PRIMARY};
        }}
        {_BTN_BASE}:disabled {{
            color: {MUTED};
            border-color: {HAIRLINE};
        }}
    """


def utility_danger_button_style():
    return f"""
        {_BTN_BASE} {{
            background: transparent;
            color: {BODY};
            border: 1px solid {HAIRLINE};
            border-radius: {ROUNDED["sm"]};
            padding: 6px 14px;
            {font_css("label")}
        }}
        {_BTN_BASE}:hover {{
            background: {rgba(ERROR_CRIMSON, 0.10)};
            color: {ERROR_CRIMSON};
            border-color: {ERROR_CRIMSON};
        }}
        {_BTN_BASE}:pressed {{
            background: {rgba(ERROR_CRIMSON, 0.22)};
        }}
        {_BTN_BASE}:disabled {{
            color: {MUTED};
            border-color: {HAIRLINE};
        }}
    """


def checkable_pill_button_style():
    return f"""
        {_BTN_BASE} {{
            background: transparent;
            color: {BODY};
            border: 1px solid {HAIRLINE};
            border-radius: {ROUNDED["sm"]};
            padding: 10px 24px;
            {font_css("button")}
        }}
        {_BTN_BASE}:hover {{
            background: {ACCENT_MUTED};
            border-color: {ACCENT};
        }}
        {_BTN_BASE}:checked {{
            background: {ACCENT};
            color: #ffffff;
            border-color: {ACCENT};
        }}
        {_BTN_BASE}:checked:hover {{
            background: {ACCENT_HOVER};
        }}
        {_BTN_BASE}:disabled {{
            color: {MUTED};
            border-color: {HAIRLINE};
        }}
    """


# ══════════════════════════════════════════════════════════════════════════════
# Panel & card styles — flat, no rounded corners
# ══════════════════════════════════════════════════════════════════════════════

def parchment_panel_style():
    return f"""
        background: {SURFACE_CARD};
        border: none;
        border-right: 1px solid {HAIRLINE};
    """


def card_style(selected=False):
    bg = ACCENT_MUTED if selected else SURFACE_CARD
    border = ACCENT if selected else HAIRLINE
    return f"""
        background: {bg};
        border: 1px solid {border};
        border-radius: {ROUNDED["sm"]};
    """


def frosted_bar_style():
    return f"""
        background: {SURFACE_NAV};
        border-bottom: 1px solid {HAIRLINE};
    """
