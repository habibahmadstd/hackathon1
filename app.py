"""
LEED v4.1/v5 IEQ Space Analyzer
================================
Automates USGBC LEED Indoor Environmental Quality (IEQ) space
categorization from architectural floor-plan images using Gemini Vision.

Author : Principal AI Engineer / Full-Stack Python Developer
Deployment : Streamlit Cloud (zero-cost tier)
AI Backend : Google Gemini 2.0 Flash via google-genai SDK
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import textwrap
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ── optional heavy imports handled gracefully ──────────────────────────────
try:
    import google.generativeai as genai  # google-genai SDK ≥ 0.8
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    import openpyxl
    from openpyxl import Workbook
    from openpyxl.styles import (
        Alignment, Border, Font, PatternFill, Side
    )
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as rl_canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from pypdf import PdfWriter
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PAGE_TITLE = "LEED IEQ Space Analyzer"
PAGE_ICON  = "🏛️"

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_FALLBACK = "gemini-3.6-flash"

# Colour palette
COLOR_REGULAR     = (34,  197,  94)   # green-500
COLOR_NON_REGULAR = (239,  68,  68)   # red-500
COLOR_UNKNOWN     = (234, 179,   8)   # yellow-500
OVERLAY_ALPHA     = 90                # 0-255 transparency for fill

# LEED classification keywords (heuristic fallback)
REGULARLY_KEYWORDS = {
    "office", "classroom", "conference", "meeting", "lounge",
    "library", "studio", "lab", "open plan", "workstation",
    "collaborative", "break room", "reception", "lobby",
}
NON_REGULAR_KEYWORDS = {
    "corridor", "hallway", "stair", "storage", "mechanical",
    "electrical", "utility", "toilet", "restroom", "bathroom",
    "server", "janitor", "closet", "loading", "parking", "ramp",
}

# ══════════════════════════════════════════════════════════════════════════════
# GEMINI SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert LEED v4.1/v5 consultant and architectural plan analyst with
deep expertise in USGBC Indoor Environmental Quality (IEQ) credits,
specifically EQ Credit: Quality Views and EQ Prerequisite: Minimum Indoor
Air Quality Performance.

## TASK
Analyze the provided architectural floor-plan image and perform the following:

1. **Scale Detection** – Identify and report the drawing scale (e.g., 1:100,
   1/8"=1'-0") from the title block, scale bar, or dimension annotations.
   If undetectable, report null.

2. **Space Inventory** – Enumerate every distinct enclosed space visible in
   the plan. For each space:
   - Extract or infer the room name/label.
   - Estimate gross floor area in square feet (SF) AND square meters (SM).
     Use the scale if available; otherwise provide a proportional estimate
     relative to the total plan footprint.
   - Classify as one of:
       • "Regularly Occupied"     – spaces where occupants spend time
         performing primary functions ≥ 1 hour continuously
         (offices, classrooms, conference rooms, open-plan work areas,
         lobbies/reception, collaborative lounges, break rooms, libraries).
       • "Non-Regularly Occupied" – spaces where occupants spend < 1 hour
         continuously or are purely transient
         (corridors, stairs, storage, mechanical/electrical rooms,
         toilets/restrooms, server rooms, loading docks, parking).
       • "Unknown" – if classification cannot be determined.
   - Provide normalized bounding-box coordinates as
     [ymin, xmin, ymax, xmax] on a 0–1000 integer scale
     where (0,0) is the TOP-LEFT corner of the image.
     These must tightly enclose the room polygon.
   - Assign a short unique room_id (e.g., "R01", "R02").

3. **Totals** – Calculate:
   - Total gross floor area (SF and SM).
   - Total regularly occupied area (SF and SM) and count.
   - Total non-regularly occupied area (SF and SM) and count.
   - Percentage of regularly occupied area relative to total.

## REASONING PROTOCOL
Before outputting JSON, reason step-by-step inside a <scratchpad> block:
- Identify the scale bar or annotation.
- List each room you can see and its approximate pixel dimensions.
- Compute pixel-to-real-world ratio.
- Classify each space with justification.
- Derive bounding boxes.

## OUTPUT FORMAT
After </scratchpad>, output ONLY a single valid JSON object — no markdown
fences, no commentary, no trailing text:

{
  "drawing_scale": "<scale string or null>",
  "scale_confidence": "<high|medium|low|unknown>",
  "total_area_sf": <number>,
  "total_area_sm": <number>,
  "regularly_occupied_area_sf": <number>,
  "regularly_occupied_area_sm": <number>,
  "regularly_occupied_count": <integer>,
  "non_regularly_occupied_area_sf": <number>,
  "non_regularly_occupied_area_sm": <number>,
  "non_regularly_occupied_count": <integer>,
  "regularly_occupied_pct": <number 0-100>,
  "spaces": [
    {
      "room_id": "R01",
      "name": "<room name>",
      "classification": "Regularly Occupied",
      "area_sf": <number>,
      "area_sm": <number>,
      "bbox_normalized": [ymin, xmin, ymax, xmax],
      "confidence": "<high|medium|low>",
      "notes": "<optional reasoning>"
    }
  ]
}
""").strip()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_api_key() -> str | None:
    """Read Gemini API key from st.secrets or environment variable."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


def _image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _extract_json(raw: str) -> dict:
    """
    Robustly extract JSON from a Gemini response that may contain
    a <scratchpad> block and/or markdown fences.
    """
    # Strip scratchpad
    raw = re.sub(r"<scratchpad>.*?</scratchpad>", "", raw, flags=re.DOTALL)
    # Strip markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "")
    raw = raw.strip()

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting the first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response:\n{raw[:500]}")


def load_image_from_upload(uploaded_file) -> Image.Image:
    """
    Convert an uploaded file (PDF or image) to a PIL Image.
    For PDFs, renders the first page at 150 DPI.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    raw_bytes = uploaded_file.read()

    if suffix == ".pdf":
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError(
                "pdf2image is not installed. "
                "Install poppler and pdf2image to process PDFs."
            )
        pages = pdf2image.convert_from_bytes(raw_bytes, dpi=150, first_page=1, last_page=1)
        return pages[0]
    else:
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI VISION CALL
# ══════════════════════════════════════════════════════════════════════════════

def analyze_floor_plan(image: Image.Image, api_key: str) -> dict:
    """
    Send floor plan to Gemini Vision and return parsed JSON analysis dict.
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-genai SDK is not installed.")

    genai.configure(api_key=api_key)

    # Try preferred model, fallback on error
    for model_name in (GEMINI_MODEL, GEMINI_FALLBACK):
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT,
            )
            # Convert image to inline Part
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            response = model.generate_content(
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": base64.b64encode(img_bytes).decode(),
                                }
                            },
                            {
                                "text": (
                                    "Analyze this architectural floor plan for LEED v4.1/v5 "
                                    "IEQ space categorization. Follow the system instructions "
                                    "exactly. Output the <scratchpad> reasoning block first, "
                                    "then the JSON object."
                                )
                            },
                        ],
                    }
                ],
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "max_output_tokens": 8192,
                },
            )
            raw_text = response.text
            result = _extract_json(raw_text)
            result["_model_used"] = model_name
            result["_raw_response"] = raw_text
            return result

        except Exception as exc:
            last_exc = exc
            continue

    raise RuntimeError(f"All Gemini models failed: {last_exc}")


# ══════════════════════════════════════════════════════════════════════════════
# EXCEL REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def build_excel_report(analysis: dict, filename: str = "floor_plan") -> bytes:
    """
    Generate a styled 2-tab Excel workbook and return raw bytes.
    Tab 1 : Executive Summary (key metrics + SUM formulas)
    Tab 2 : Space Detail Table (color-coded by classification)
    """
    if not OPENPYXL_AVAILABLE:
        raise RuntimeError("openpyxl is not installed.")

    wb = Workbook()

    # ── Colour palette ────────────────────────────────────────────────────
    GREEN_DARK   = "1A7A3C"
    GREEN_MID    = "22C55E"
    GREEN_LIGHT  = "DCFCE7"
    RED_DARK     = "991B1B"
    RED_MID      = "EF4444"
    RED_LIGHT    = "FEE2E2"
    YELLOW_LIGHT = "FEF9C3"
    GREY_HEADER  = "1E293B"   # slate-900
    GREY_ALT     = "F8FAFC"   # slate-50
    WHITE        = "FFFFFF"
    ACCENT       = "0F172A"   # slate-950

    def _fill(hex_color: str) -> PatternFill:
        return PatternFill("solid", fgColor=hex_color)

    def _font(bold=False, color=ACCENT, size=11) -> Font:
        return Font(name="Calibri", bold=bold, color=color, size=size)

    def _border() -> Border:
        s = Side(style="thin", color="D1D5DB")
        return Border(left=s, right=s, top=s, bottom=s)

    def _center() -> Alignment:
        return Alignment(horizontal="center", vertical="center", wrap_text=True)

    def _left() -> Alignment:
        return Alignment(horizontal="left", vertical="center", wrap_text=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 – Executive Summary
    # ══════════════════════════════════════════════════════════════════════
    ws1 = wb.active
    ws1.title = "Executive Summary"
    ws1.sheet_view.showGridLines = False

    # Column widths
    for col, w in enumerate([3, 35, 22, 22, 22, 3], start=1):
        ws1.column_dimensions[get_column_letter(col)].width = w

    # Row heights
    ws1.row_dimensions[1].height  = 8
    ws1.row_dimensions[2].height  = 45
    ws1.row_dimensions[3].height  = 8

    # Title bar
    ws1.merge_cells("B2:E2")
    title_cell = ws1["B2"]
    title_cell.value       = f"LEED v4.1/v5 IEQ Space Analysis — {filename}"
    title_cell.font        = Font(name="Calibri", bold=True, size=18, color=WHITE)
    title_cell.fill        = _fill(GREY_HEADER)
    title_cell.alignment   = _center()

    # Section headers row
    row = 4
    for col, label in enumerate(
        ["Metric", "Value (Imperial)", "Value (Metric)", "Notes"], start=2
    ):
        c = ws1.cell(row=row, column=col, value=label)
        c.font      = _font(bold=True, color=WHITE, size=11)
        c.fill      = _fill(GREY_HEADER)
        c.alignment = _center()
        c.border    = _border()
    ws1.row_dimensions[row].height = 22

    # Data rows
    metrics = [
        ("Drawing Scale",          analysis.get("drawing_scale") or "—",
         "",                       f"Confidence: {analysis.get('scale_confidence','—')}"),
        ("Total Floor Area",       f"{analysis.get('total_area_sf',0):,.0f} SF",
         f"{analysis.get('total_area_sm',0):,.1f} m²",  "Gross area"),
        ("Regularly Occupied Area",f"{analysis.get('regularly_occupied_area_sf',0):,.0f} SF",
         f"{analysis.get('regularly_occupied_area_sm',0):,.1f} m²",
         f"{analysis.get('regularly_occupied_count',0)} spaces"),
        ("Non-Regularly Occ. Area",f"{analysis.get('non_regularly_occupied_area_sf',0):,.0f} SF",
         f"{analysis.get('non_regularly_occupied_area_sm',0):,.1f} m²",
         f"{analysis.get('non_regularly_occupied_count',0)} spaces"),
        ("% Regularly Occupied",   f"{analysis.get('regularly_occupied_pct',0):.1f}%",
         "",                       "Of total gross area"),
    ]

    fill_cycle = [_fill(WHITE), _fill(GREY_ALT)]
    for i, (metric, val_imp, val_met, notes) in enumerate(metrics):
        r = row + 1 + i
        ws1.row_dimensions[r].height = 22
        row_fill = fill_cycle[i % 2]

        # Highlight regulatory threshold row
        if "%" in val_imp:
            pct = analysis.get("regularly_occupied_pct", 0)
            row_fill = _fill(GREEN_LIGHT) if pct >= 50 else _fill(RED_LIGHT)

        for col, val in enumerate([metric, val_imp, val_met, notes], start=2):
            c = ws1.cell(row=r, column=col, value=val)
            c.fill      = row_fill
            c.font      = _font(bold=(col == 2))
            c.alignment = _center() if col > 2 else _left()
            c.border    = _border()

    # Spacer
    note_row = row + len(metrics) + 2
    ws1.row_dimensions[note_row].height = 16
    ws1.merge_cells(f"B{note_row}:E{note_row}")
    nc = ws1[f"B{note_row}"]
    nc.value     = "⚠  Areas are AI-estimated. Verify against stamped construction documents before LEED submission."
    nc.font      = Font(name="Calibri", italic=True, size=9, color="6B7280")
    nc.alignment = _left()

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 – Space Detail
    # ══════════════════════════════════════════════════════════════════════
    ws2 = wb.create_sheet("Space Detail")
    ws2.sheet_view.showGridLines = False

    col_defs = [
        ("Room ID",        12),
        ("Room Name",      30),
        ("Classification", 26),
        ("Area (SF)",      14),
        ("Area (m²)",      14),
        ("Confidence",     14),
        ("Notes",          40),
    ]
    for col, (label, width) in enumerate(col_defs, start=1):
        ws2.column_dimensions[get_column_letter(col)].width = width

    # Header
    ws2.row_dimensions[1].height = 10
    ws2.row_dimensions[2].height = 24
    for col, (label, _) in enumerate(col_defs, start=1):
        c = ws2.cell(row=2, column=col, value=label)
        c.font      = _font(bold=True, color=WHITE, size=11)
        c.fill      = _fill(GREY_HEADER)
        c.alignment = _center()
        c.border    = _border()

    spaces = analysis.get("spaces", [])
    for i, space in enumerate(spaces):
        r = 3 + i
        ws2.row_dimensions[r].height = 20
        cls = space.get("classification", "Unknown")

        if "Regularly" in cls:
            row_fill = _fill(GREEN_LIGHT)
        elif "Non-Regularly" in cls:
            row_fill = _fill(RED_LIGHT)
        else:
            row_fill = _fill(YELLOW_LIGHT)

        values = [
            space.get("room_id", f"R{i+1:02d}"),
            space.get("name", "—"),
            cls,
            round(space.get("area_sf", 0), 1),
            round(space.get("area_sm", 0), 2),
            space.get("confidence", "—"),
            space.get("notes", ""),
        ]
        for col, val in enumerate(values, start=1):
            c = ws2.cell(row=r, column=col, value=val)
            c.fill      = row_fill
            c.font      = _font()
            c.alignment = _center() if col != 2 else _left()
            c.border    = _border()

    # Totals row with Excel SUM formulas
    total_row = 3 + len(spaces)
    ws2.row_dimensions[total_row].height = 22
    total_fill = _fill(GREY_HEADER)
    for col in range(1, len(col_defs) + 1):
        c = ws2.cell(row=total_row, column=col)
        c.fill   = total_fill
        c.border = _border()

    ws2.cell(total_row, 1, "TOTAL").font = _font(bold=True, color=WHITE)
    ws2.cell(total_row, 2, f"{len(spaces)} spaces").font = _font(color=WHITE)
    sf_col  = get_column_letter(4)
    sm_col  = get_column_letter(5)
    data_start = 3
    data_end   = 2 + len(spaces)
    ws2.cell(total_row, 4,
             f"=SUM({sf_col}{data_start}:{sf_col}{data_end})").font = _font(bold=True, color=WHITE)
    ws2.cell(total_row, 5,
             f"=SUM({sm_col}{data_start}:{sm_col}{data_end})").font = _font(bold=True, color=WHITE)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
# PDF ANNOTATION RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def annotate_floor_plan(
    base_image: Image.Image,
    analysis: dict,
) -> bytes:
    """
    Draw semi-transparent colour overlays and callout badge labels onto the
    floor plan, then encode the result as a single-page PDF.
    Returns raw PDF bytes.
    """
    img = base_image.convert("RGBA")
    W, H = img.size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    # Attempt to load a proportional font; fall back to default
    try:
        badge_font   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        label_font   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        badge_font = label_font = ImageFont.load_default()

    spaces = analysis.get("spaces", [])

    for space in spaces:
        bbox = space.get("bbox_normalized")
        if not bbox or len(bbox) != 4:
            continue

        ymin_n, xmin_n, ymax_n, xmax_n = bbox
        # Convert 0-1000 to pixel coords
        x0 = int(xmin_n / 1000 * W)
        y0 = int(ymin_n / 1000 * H)
        x1 = int(xmax_n / 1000 * W)
        y1 = int(ymax_n / 1000 * H)

        if x1 <= x0 or y1 <= y0:
            continue

        cls = space.get("classification", "Unknown")
        if "Regularly" in cls and "Non" not in cls:
            fill_rgba  = (*COLOR_REGULAR,     OVERLAY_ALPHA)
            badge_rgba = (*COLOR_REGULAR,     220)
            label_text = "REG"
        elif "Non-Regularly" in cls:
            fill_rgba  = (*COLOR_NON_REGULAR, OVERLAY_ALPHA)
            badge_rgba = (*COLOR_NON_REGULAR, 220)
            label_text = "NON"
        else:
            fill_rgba  = (*COLOR_UNKNOWN,     OVERLAY_ALPHA)
            badge_rgba = (*COLOR_UNKNOWN,     220)
            label_text = "UNK"

        # Semi-transparent fill
        draw.rectangle([x0, y0, x1, y1], fill=fill_rgba)
        # Outline
        draw.rectangle([x0, y0, x1, y1], outline=(*badge_rgba[:3], 255), width=2)

        # Badge dimensions
        room_id = space.get("room_id", "")
        badge_text = f"{room_id} {label_text}"
        bw, bh = 90, 22
        bx = x0 + 4
        by = y0 + 4
        # Clamp to image bounds
        bx = min(bx, W - bw - 4)
        by = min(by, H - bh - 4)

        # Badge background
        draw.rounded_rectangle([bx, by, bx+bw, by+bh],
                                radius=4, fill=badge_rgba)
        # Badge text
        draw.text((bx+5, by+4), badge_text, font=badge_font, fill=(255, 255, 255, 255))

        # Room name below badge
        name = space.get("name", "")[:22]
        area = f"{space.get('area_sf',0):.0f} SF"
        draw.text((bx, by+bh+2), name, font=label_font, fill=(*badge_rgba[:3], 230))
        draw.text((bx, by+bh+14), area, font=label_font, fill=(*badge_rgba[:3], 200))

    # Merge overlay onto base
    merged = Image.alpha_composite(img, overlay).convert("RGB")

    if not REPORTLAB_AVAILABLE:
        # Fallback: return plain PNG wrapped in PDF via Pillow save
        buf = io.BytesIO()
        merged.save(buf, format="PDF")
        buf.seek(0)
        return buf.read()

    # Build PDF with ReportLab
    img_buf = io.BytesIO()
    merged.save(img_buf, format="PNG", optimize=False)
    img_buf.seek(0)

    pdf_buf = io.BytesIO()
    page_w  = W * mm / 3.7795          # pixels → mm (96 dpi)
    page_h  = H * mm / 3.7795
    c       = rl_canvas.Canvas(pdf_buf, pagesize=(page_w, page_h))

    c.drawInlineImage(merged, 0, 0, width=page_w, height=page_h)

    # Legend box
    leg_x, leg_y = 10 * mm, 10 * mm
    leg_w, leg_h = 52 * mm, 28 * mm
    c.setFillColorRGB(1, 1, 1, alpha=0.85)
    c.roundRect(leg_x, leg_y, leg_w, leg_h, radius=3 * mm, fill=1, stroke=0)

    c.setFont("Helvetica-Bold", 7)
    c.setFillColorRGB(0.12, 0.12, 0.12)
    c.drawString(leg_x + 3*mm, leg_y + 22*mm, "LEED IEQ SPACE LEGEND")

    dot_size = 3 * mm
    for row_i, (label, rgb) in enumerate([
        ("Regularly Occupied",      [r/255 for r in COLOR_REGULAR]),
        ("Non-Regularly Occupied",  [r/255 for r in COLOR_NON_REGULAR]),
        ("Unknown",                 [r/255 for r in COLOR_UNKNOWN]),
    ]):
        ry = leg_y + 16*mm - row_i * 5*mm
        c.setFillColorRGB(*rgb)
        c.rect(leg_x + 3*mm, ry, dot_size, dot_size, fill=1, stroke=0)
        c.setFillColorRGB(0.12, 0.12, 0.12)
        c.setFont("Helvetica", 6.5)
        c.drawString(leg_x + 8*mm, ry + 0.8*mm, label)

    c.save()
    pdf_buf.seek(0)
    return pdf_buf.read()


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

def _metric_card(label: str, value: str, delta: str = "", color: str = "#22C55E") -> str:
    return f"""
    <div style="
        background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px;
        padding:18px 20px; box-shadow:0 1px 3px rgba(0,0,0,.06);
        border-top:4px solid {color}; min-height:100px;">
      <p style="margin:0; font-size:12px; color:#64748B; font-weight:600;
                letter-spacing:.04em; text-transform:uppercase;">{label}</p>
      <p style="margin:4px 0 0; font-size:28px; font-weight:700;
                color:#0F172A; line-height:1.15;">{value}</p>
      <p style="margin:2px 0 0; font-size:12px; color:#94A3B8;">{delta}</p>
    </div>"""


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Global styles ──────────────────────────────────────────────────────
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      .block-container { padding-top: 2rem; max-width: 1280px; }
      div[data-testid="stDownloadButton"] button {
          background: #1E293B; color: #FFFFFF; border: none;
          border-radius: 8px; font-weight: 600; padding: 10px 20px;
          transition: background .2s;
      }
      div[data-testid="stDownloadButton"] button:hover { background: #334155; }
      .section-header {
          font-size: 13px; font-weight: 700; color: #64748B;
          letter-spacing: .08em; text-transform: uppercase;
          margin: 1.5rem 0 .5rem; border-bottom: 1px solid #E2E8F0;
          padding-bottom: .4rem;
      }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/"
            "USGBC_logo.svg/320px-USGBC_logo.svg.png",
            width=160,
        )
        st.markdown("### LEED IEQ Analyzer")
        st.caption(
            "Automates USGBC LEED v4.1/v5 Indoor Environmental Quality "
            "space categorization from architectural floor plans using "
            "Gemini Vision AI."
        )
        st.divider()

        api_key = _get_api_key()
        if api_key:
            st.success("✓ API key loaded", icon="🔑")
        else:
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                help="Get a free key at aistudio.google.com",
                placeholder="AIza…",
            )
            if api_key:
                st.success("Key accepted for this session")

        st.divider()
        st.markdown(
            "**Model:** `gemini-2.5-flash`  \n"
            "**Standard:** LEED v4.1/v5 IEQ  \n"
            "**Output:** Excel + Annotated PDF"
        )
        st.divider()
        st.caption(
            "Area values are AI-estimated. Always verify against stamped "
            "construction documents before LEED submission."
        )

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='font-size:32px;font-weight:800;color:#0F172A;"
        "margin-bottom:4px;'>🏛️ LEED IEQ Space Analyzer</h1>"
        "<p style='color:#64748B;font-size:15px;margin-top:0;'>"
        "Upload an architectural floor plan → AI classifies every space → "
        "Download Excel report + Annotated PDF</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── File Upload ────────────────────────────────────────────────────────
    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<p class="section-header">01 · Upload Floor Plan</p>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            label="Drag & drop or click to browse",
            type=["pdf", "png", "jpg", "jpeg"],
            help="PDF (first page), PNG, JPG accepted. Max 200 MB.",
        )

    with col_preview:
        st.markdown('<p class="section-header">02 · Preview</p>',
                    unsafe_allow_html=True)
        preview_placeholder = st.empty()

    pil_image: Image.Image | None = None

    if uploaded:
        with st.spinner("Loading file…"):
            try:
                uploaded.seek(0)
                pil_image = load_image_from_upload(uploaded)
                with col_preview:
                    preview_placeholder.image(
                        pil_image,
                        caption=f"{uploaded.name}  "
                                f"({pil_image.width}×{pil_image.height} px)",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Could not load file: {e}")
                pil_image = None

    # ── Analysis Button ────────────────────────────────────────────────────
    st.markdown('<p class="section-header">03 · Analyze</p>',
                unsafe_allow_html=True)

    run_disabled = pil_image is None or not api_key
    run_btn = st.button(
        "🔍  Analyze Floor Plan",
        disabled=run_disabled,
        type="primary",
        use_container_width=False,
    )
    if run_disabled and not run_btn:
        if pil_image is None:
            st.info("Upload a floor plan to continue.")
        elif not api_key:
            st.warning("Enter your Gemini API key in the sidebar.")

    # ── Results ────────────────────────────────────────────────────────────
    if run_btn and pil_image and api_key:
        with st.spinner("Analyzing floor plan with Gemini Vision… (may take 15-45 s)"):
            try:
                analysis = analyze_floor_plan(pil_image, api_key)
                st.session_state["analysis"]  = analysis
                st.session_state["pil_image"] = pil_image
                st.session_state["filename"]  = Path(uploaded.name).stem
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
                st.stop()

    if "analysis" in st.session_state:
        analysis  = st.session_state["analysis"]
        pil_image = st.session_state["pil_image"]
        stem      = st.session_state.get("filename", "floor_plan")

        st.divider()
        st.markdown('<p class="section-header">04 · Results</p>',
                    unsafe_allow_html=True)

        # Metric cards
        pct = analysis.get("regularly_occupied_pct", 0)
        cards_html = "".join([
            _metric_card(
                "Total Floor Area",
                f"{analysis.get('total_area_sf',0):,.0f} SF",
                f"{analysis.get('total_area_sm',0):,.1f} m²",
            ),
            _metric_card(
                "Regularly Occupied",
                f"{analysis.get('regularly_occupied_area_sf',0):,.0f} SF",
                f"{analysis.get('regularly_occupied_count',0)} spaces · "
                f"{analysis.get('regularly_occupied_area_sm',0):,.1f} m²",
                color="#22C55E",
            ),
            _metric_card(
                "Non-Regularly Occupied",
                f"{analysis.get('non_regularly_occupied_area_sf',0):,.0f} SF",
                f"{analysis.get('non_regularly_occupied_count',0)} spaces · "
                f"{analysis.get('non_regularly_occupied_area_sm',0):,.1f} m²",
                color="#EF4444",
            ),
            _metric_card(
                "% Regularly Occupied",
                f"{pct:.1f}%",
                "of gross floor area",
                color="#22C55E" if pct >= 50 else "#EF4444",
            ),
        ])

        col1, col2, col3, col4 = st.columns(4)
        for col, card in zip([col1, col2, col3, col4], [
            _metric_card(
                "Total Floor Area",
                f"{analysis.get('total_area_sf',0):,.0f} SF",
                f"{analysis.get('total_area_sm',0):,.1f} m²",
                "#6366F1",
            ),
            _metric_card(
                "Regularly Occupied",
                f"{analysis.get('regularly_occupied_area_sf',0):,.0f} SF",
                f"{analysis.get('regularly_occupied_count',0)} spaces",
                "#22C55E",
            ),
            _metric_card(
                "Non-Regularly Occupied",
                f"{analysis.get('non_regularly_occupied_area_sf',0):,.0f} SF",
                f"{analysis.get('non_regularly_occupied_count',0)} spaces",
                "#EF4444",
            ),
            _metric_card(
                "% Regularly Occupied",
                f"{pct:.1f}%",
                "of gross floor area",
                "#22C55E" if pct >= 50 else "#EF4444",
            ),
        ]):
            col.markdown(card, unsafe_allow_html=True)

        st.write("")

        # Drawing scale info
        if analysis.get("drawing_scale"):
            st.info(
                f"📐  Drawing Scale: **{analysis['drawing_scale']}**  "
                f"(confidence: {analysis.get('scale_confidence','—')})"
            )

        # Space detail table
        spaces = analysis.get("spaces", [])
        if spaces:
            st.markdown('<p class="section-header">Space Detail Table</p>',
                        unsafe_allow_html=True)
            df = pd.DataFrame([
                {
                    "ID":             s.get("room_id", ""),
                    "Name":           s.get("name", ""),
                    "Classification": s.get("classification", ""),
                    "Area (SF)":      round(s.get("area_sf", 0), 1),
                    "Area (m²)":      round(s.get("area_sm", 0), 2),
                    "Confidence":     s.get("confidence", ""),
                    "Notes":          s.get("notes", ""),
                }
                for s in spaces
            ])

            def _color_cls(val: str) -> str:
                if "Non-Regularly" in val:
                    return "background-color:#FEE2E2; color:#991B1B; font-weight:600"
                if "Regularly" in val:
                    return "background-color:#DCFCE7; color:#166534; font-weight:600"
                return "background-color:#FEF9C3; color:#854D0E; font-weight:600"

            styled = df.style.applymap(_color_cls, subset=["Classification"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── Downloads ──────────────────────────────────────────────────────
        st.divider()
        st.markdown('<p class="section-header">05 · Download Reports</p>',
                    unsafe_allow_html=True)
        dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 2])

        with dl_col1:
            with st.spinner("Building Excel report…"):
                try:
                    excel_bytes = build_excel_report(analysis, filename=stem)
                    st.download_button(
                        label="📊  Download Excel Report",
                        data=excel_bytes,
                        file_name=f"{stem}_LEED_IEQ_Report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument"
                              ".spreadsheetml.sheet",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Excel error: {e}")

        with dl_col2:
            with st.spinner("Rendering annotated PDF…"):
                try:
                    pdf_bytes = annotate_floor_plan(pil_image, analysis)
                    st.download_button(
                        label="📄  Download Annotated PDF",
                        data=pdf_bytes,
                        file_name=f"{stem}_LEED_Annotated.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"PDF error: {e}")

        # Annotated preview
        st.markdown('<p class="section-header">Annotated Preview</p>',
                    unsafe_allow_html=True)
        try:
            annotated_img = _render_preview(pil_image, analysis)
            st.image(annotated_img, use_container_width=True,
                     caption="Green = Regularly Occupied · Red = Non-Regularly Occupied · Yellow = Unknown")
        except Exception as e:
            st.warning(f"Preview unavailable: {e}")

        # Raw JSON expander
        with st.expander("🔍  View raw Gemini response JSON"):
            raw = analysis.pop("_raw_response", None)
            st.json(analysis)
            if raw:
                analysis["_raw_response"] = raw
                st.markdown("**Full model response:**")
                st.text_area("", raw, height=300, label_visibility="collapsed")


def _render_preview(base_image: Image.Image, analysis: dict) -> Image.Image:
    """Lightweight in-app preview (no PDF, returns PIL Image)."""
    img = base_image.copy().convert("RGBA")
    W, H = img.size
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13
        )
    except Exception:
        font = ImageFont.load_default()

    for space in analysis.get("spaces", []):
        bbox = space.get("bbox_normalized")
        if not bbox or len(bbox) != 4:
            continue
        ymin_n, xmin_n, ymax_n, xmax_n = bbox
        x0 = int(xmin_n / 1000 * W)
        y0 = int(ymin_n / 1000 * H)
        x1 = int(xmax_n / 1000 * W)
        y1 = int(ymax_n / 1000 * H)
        if x1 <= x0 or y1 <= y0:
            continue

        cls = space.get("classification", "Unknown")
        if "Non" in cls:
            col = (*COLOR_NON_REGULAR, OVERLAY_ALPHA)
            bcol = (*COLOR_NON_REGULAR, 210)
        elif "Regularly" in cls:
            col = (*COLOR_REGULAR, OVERLAY_ALPHA)
            bcol = (*COLOR_REGULAR, 210)
        else:
            col = (*COLOR_UNKNOWN, OVERLAY_ALPHA)
            bcol = (*COLOR_UNKNOWN, 210)

        draw.rectangle([x0, y0, x1, y1], fill=col, outline=(*bcol[:3], 255), width=2)
        label = f"{space.get('room_id','')} {space.get('name','')[:14]}"
        draw.rectangle([x0+2, y0+2, x0+len(label)*8+6, y0+18], fill=bcol)
        draw.text((x0+4, y0+3), label, font=font, fill=(255, 255, 255, 255))

    return Image.alpha_composite(img, overlay).convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
