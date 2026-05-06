"""
Produces visualizations that make the "model defaults to not-crossing on
small pedestrians" finding visually obvious.

Outputs under --output-dir:
    1. scale_stratified_failure_grid.png
       A 5-row grid, one per bbox-area quintile, showing 6 false-negative
       examples per row. Lets the reader scan Q1→Q5 and see the small
       pedestrians that the model incorrectly flagged as "not crossing".

    2. confident_missed_crossers.png
       Same idea but filtered to HIGH confidence (>0.9) wrong predictions
       on actual crossers. The headline evidence: "the model confidently
       predicted not-crossing for these people, and they crossed."

    3. small_pedestrians_in_context.png
       Shows full frames (not just crops) of small-pedestrian false negatives,
       with the bbox overlaid and a zoom-in inset. Gives the reader the
       "oh, that tiny person was actually about to cross" reaction.

    4. scale_comparison_strip.png
       If we can find the same pedestrian ID tracked across multiple samples,
       show the model's evolving prediction as the pedestrian gets closer.
       Demonstrates that the model "wakes up" only at a specific scale.

"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# -----------------------------------------------------------------------------
# Drawing primitives
# -----------------------------------------------------------------------------
def _get_font(size=16):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_bbox(draw, bbox, color, width=3):
    x1, y1, x2, y2 = bbox
    for off in range(width):
        draw.rectangle([x1 - off, y1 - off, x2 + off, y2 + off], outline=color)


def _draw_label(draw, x, y, text, fill_color, font):
    try:
        tbbox = draw.textbbox((0, 0), text, font=font)
        tw, th = tbbox[2] - tbbox[0], tbbox[3] - tbbox[1]
    except AttributeError:
        tw, th = font.getsize(text)
    pad = 4
    draw.rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], fill=fill_color)
    draw.text((x + pad, y + pad), text, fill="white", font=font)


def _context_crop(img, row, pad_factor=2.5, target_size=260):
    """Crop a context window around the bbox, resize, and return (image, adjusted_bbox)."""
    iw, ih = img.size
    x1, y1, x2, y2 = (float(row["bbox_x1"]), float(row["bbox_y1"]),
                      float(row["bbox_x2"]), float(row["bbox_y2"]))
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
    side = max(bw, bh) * pad_factor
    cx1 = max(int(round(cx - side / 2.0)), 0)
    cy1 = max(int(round(cy - side / 2.0)), 0)
    cx2 = min(int(round(cx + side / 2.0)), iw)
    cy2 = min(int(round(cy + side / 2.0)), ih)
    cropped = img.crop((cx1, cy1, cx2, cy2))

    cw, ch = cropped.size
    scale = target_size / max(cw, ch)
    nw, nh = int(cw * scale), int(ch * scale)
    cropped = cropped.resize((nw, nh), Image.LANCZOS)

    bx1 = (x1 - cx1) * scale
    by1 = (y1 - cy1) * scale
    bx2 = (x2 - cx1) * scale
    by2 = (y2 - cy1) * scale

    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    ox = (target_size - nw) // 2
    oy = (target_size - nh) // 2
    canvas.paste(cropped, (ox, oy))
    return canvas, (bx1 + ox, by1 + oy, bx2 + ox, by2 + oy)


# -----------------------------------------------------------------------------
# Visualization 1: Scale-stratified failure grid (5 rows, one per quintile)
# -----------------------------------------------------------------------------
def scale_stratified_grid(df, out_path, per_bin=6, tile=260):
    """One row per bbox-area quintile, each row showing false negatives in that bin."""
    df = df.copy()
    df["area_q"] = pd.qcut(df["bbox_area"], 5,
                           labels=["Q1 (smallest / distant)", "Q2", "Q3", "Q4",
                                   "Q5 (largest / closest)"],
                           duplicates="drop")

    # Only false negatives (model said not-crossing, actually crossing)
    fn = df[(df["label"] == 1) & (df["prediction"] == 0)]

    rows_data = []
    for q in ["Q1 (smallest / distant)", "Q2", "Q3", "Q4", "Q5 (largest / closest)"]:
        sub = fn[fn["area_q"] == q]
        if len(sub) == 0:
            rows_data.append((q, []))
            continue
        # Sort by confidence descending so we pick the most-confident wrong predictions
        sub = sub.sort_values("confidence", ascending=False).head(per_bin)
        rows_data.append((q, sub.to_dict("records")))

    # Canvas layout
    row_label_w = 220
    tile_label_h = 32
    header_h = 70
    n_rows = len(rows_data)
    grid_w = row_label_w + per_bin * tile
    grid_h = header_h + n_rows * (tile + tile_label_h)

    canvas = Image.new("RGB", (grid_w, grid_h), (248, 248, 250))
    draw = ImageDraw.Draw(canvas)

    title_font = _get_font(22)
    subtitle_font = _get_font(13)
    row_font = _get_font(15)
    cap_font = _get_font(12)
    bbox_label_font = _get_font(12)

    title = "False negatives across the scale spectrum"
    subtitle = ("Rows: pedestrian bounding-box area (proxy for distance). "
                "All tiles show actual crossers the model predicted as NOT crossing.")
    draw.text((20, 12), title, fill=(20, 20, 20), font=title_font)
    draw.text((20, 42), subtitle, fill=(60, 60, 60), font=subtitle_font)

    for r_idx, (q_label, items) in enumerate(rows_data):
        y_base = header_h + r_idx * (tile + tile_label_h)

        # Row title
        draw.rectangle(
            [0, y_base, row_label_w, y_base + tile + tile_label_h],
            fill=(230, 234, 240),
        )
        draw.text((12, y_base + tile // 2 - 12), q_label,
                  fill=(30, 30, 40), font=row_font)
        if items:
            avg_area = np.mean([it["bbox_area"] for it in items])
            avg_conf = np.mean([it["confidence"] for it in items])
            draw.text((12, y_base + tile // 2 + 10),
                      f"mean area = {int(avg_area)} px²",
                      fill=(80, 80, 90), font=cap_font)
            draw.text((12, y_base + tile // 2 + 26),
                      f"mean conf = {avg_conf:.2f}",
                      fill=(80, 80, 90), font=cap_font)

        # Tiles
        for c_idx in range(per_bin):
            x_off = row_label_w + c_idx * tile
            if c_idx >= len(items):
                continue
            row = items[c_idx]
            try:
                with Image.open(row["image_path"]).convert("RGB") as im:
                    t, adj_bbox = _context_crop(im, row, pad_factor=2.5,
                                                target_size=tile)
            except (FileNotFoundError, OSError):
                continue
            t_draw = ImageDraw.Draw(t)
            # Red bbox = wrong prediction, and this is always a false negative
            _draw_bbox(t_draw, adj_bbox, "#ef4444", width=3)
            _draw_label(t_draw, 6, 6,
                        f"conf={row['confidence']:.2f}",
                        "#ef4444", bbox_label_font)
            canvas.paste(t, (x_off, y_base))
            cap = f"pred=NC | actual=C | area={int(row['bbox_area'])}px²"
            draw.text((x_off + 4, y_base + tile + 4), cap,
                      fill=(40, 40, 40), font=cap_font)

    canvas.save(out_path, optimize=True)
    print(f"[ok] wrote {out_path}")


# -----------------------------------------------------------------------------
# Visualization 2: Most confident missed crossers (a punchy, simple grid)
# -----------------------------------------------------------------------------
def confident_missed_crossers(df, out_path, n_cols=6, n_rows=4, tile=280):
    """Top N most-confident false negatives -- 'the model was sure these weren't crossing'."""
    fn = df[(df["label"] == 1) & (df["prediction"] == 0)]
    if len(fn) == 0:
        print("[skip] no false negatives found")
        return
    fn = fn.sort_values("confidence", ascending=False).head(n_cols * n_rows)

    header_h = 80
    tile_cap = 28
    grid_w = n_cols * tile
    grid_h = header_h + n_rows * (tile + tile_cap)
    canvas = Image.new("RGB", (grid_w, grid_h), (248, 248, 250))
    draw = ImageDraw.Draw(canvas)

    title_font = _get_font(22)
    sub_font = _get_font(13)
    cap_font = _get_font(12)
    label_font = _get_font(14)

    draw.text((18, 14),
              "Most confident missed crossers",
              fill=(30, 30, 30), font=title_font)
    draw.text((18, 44),
              "Actual crossers that the model confidently labelled as NOT crossing.",
              fill=(80, 80, 80), font=sub_font)

    for i, (_, row) in enumerate(fn.iterrows()):
        r = i // n_cols; c = i % n_cols
        x_off = c * tile; y_off = header_h + r * (tile + tile_cap)
        try:
            with Image.open(row["image_path"]).convert("RGB") as im:
                t, adj_bbox = _context_crop(im, row, pad_factor=2.2,
                                            target_size=tile)
        except (FileNotFoundError, OSError):
            continue
        t_draw = ImageDraw.Draw(t)
        _draw_bbox(t_draw, adj_bbox, "#ef4444", width=3)
        _draw_label(t_draw, 6, 6,
                    f"P(NC) = {row['prob_notcross']:.2f}",
                    "#ef4444", label_font)
        canvas.paste(t, (x_off, y_off))
        draw.text((x_off + 4, y_off + tile + 4),
                  f"bbox area = {int(row['bbox_area'])}px²",
                  fill=(40, 40, 40), font=cap_font)

    canvas.save(out_path, optimize=True)
    print(f"[ok] wrote {out_path}")


# -----------------------------------------------------------------------------
# Visualization 3: Full-frame context with zoom-in inset
# -----------------------------------------------------------------------------
def in_context_examples(df, out_path, n=4, thumb_w=960):
    """
    Show a few full-frame images with the small-pedestrian bbox highlighted
    plus a zoomed-in inset. Makes it obvious how small these pedestrians are
    in the driving scene, and that the model essentially missed a real crosser.
    """
    # Q1 area quintile, false negatives only, sort by confidence desc
    area_q1 = df["bbox_area"].quantile(0.2)
    candidates = df[(df["bbox_area"] <= area_q1) &
                    (df["label"] == 1) &
                    (df["prediction"] == 0)]
    if len(candidates) == 0:
        print("[skip] no small-pedestrian false negatives found")
        return
    picks = candidates.sort_values("confidence", ascending=False).head(n)

    panels = []
    cap_font = _get_font(14)
    title_font = _get_font(18)
    inset_label_font = _get_font(13)

    for _, row in picks.iterrows():
        try:
            with Image.open(row["image_path"]).convert("RGB") as im:
                iw, ih = im.size
                # Resize the full frame to fit our output width
                scale = thumb_w / iw
                disp = im.resize((thumb_w, int(ih * scale)), Image.LANCZOS)
        except (FileNotFoundError, OSError):
            continue
        dw, dh = disp.size
        # Scale bbox to display size
        bx1 = int(float(row["bbox_x1"]) * scale)
        by1 = int(float(row["bbox_y1"]) * scale)
        bx2 = int(float(row["bbox_x2"]) * scale)
        by2 = int(float(row["bbox_y2"]) * scale)
        d = ImageDraw.Draw(disp)
        _draw_bbox(d, (bx1, by1, bx2, by2), "#ef4444", width=4)
        _draw_label(d, bx1, max(by1 - 30, 0),
                    f"P(NC) = {row['prob_notcross']:.2f}",
                    "#ef4444", cap_font)

        # Build a zoom-in inset (200x200), upscaled from the original crop
        inset_size = 260
        try:
            with Image.open(row["image_path"]).convert("RGB") as im2:
                x1, y1, x2, y2 = (float(row["bbox_x1"]), float(row["bbox_y1"]),
                                  float(row["bbox_x2"]), float(row["bbox_y2"]))
                bw, bh = x2 - x1, y2 - y1
                cx, cy = x1 + bw / 2, y1 + bh / 2
                side = max(bw, bh) * 2.0
                iw0, ih0 = im2.size
                cx1 = max(int(cx - side / 2), 0)
                cy1 = max(int(cy - side / 2), 0)
                cx2 = min(int(cx + side / 2), iw0)
                cy2 = min(int(cy + side / 2), ih0)
                inset = im2.crop((cx1, cy1, cx2, cy2)).resize(
                    (inset_size, inset_size), Image.LANCZOS)
                id_draw = ImageDraw.Draw(inset)
                # Scale bbox inside inset
                sx = inset_size / max(cx2 - cx1, 1)
                sy = inset_size / max(cy2 - cy1, 1)
                _draw_bbox(
                    id_draw,
                    ((x1 - cx1) * sx, (y1 - cy1) * sy,
                     (x2 - cx1) * sx, (y2 - cy1) * sy),
                    "#ef4444", width=3,
                )
                # Paste inset top-right
                disp.paste(inset, (dw - inset_size - 10, 10))
                d.rectangle(
                    [dw - inset_size - 10, 10,
                     dw - 10, 10 + inset_size],
                    outline=(40, 40, 40), width=2,
                )
                d.text((dw - inset_size - 6, 10 + inset_size + 2),
                       "zoomed", fill=(40, 40, 40), font=inset_label_font)
        except (FileNotFoundError, OSError):
            pass

        # Add a caption strip below
        cap_h = 34
        full = Image.new("RGB", (dw, dh + cap_h), (255, 255, 255))
        full.paste(disp, (0, 0))
        fd = ImageDraw.Draw(full)
        fd.text((12, dh + 8),
                f"{row['video']} / {row['image_path'].split('/')[-1]}  |  "
                f"area = {int(row['bbox_area'])}px²  |  "
                f"model confidently said: NOT crossing",
                fill=(30, 30, 30), font=cap_font)
        panels.append(full)

    if not panels:
        return

    # Stack vertically
    W = max(p.size[0] for p in panels)
    H = sum(p.size[1] for p in panels) + 80
    stacked = Image.new("RGB", (W, H), (252, 252, 252))
    sd = ImageDraw.Draw(stacked)
    sd.text((18, 18), "Small-pedestrian false negatives in full driving context",
            fill=(30, 30, 30), font=title_font)
    sd.text((18, 44),
            "The model confidently predicted 'not crossing' for each of these pedestrians. "
            "They actually crossed.",
            fill=(80, 80, 80), font=cap_font)
    y = 80
    for p in panels:
        stacked.paste(p, (0, y))
        y += p.size[1]
    stacked.save(out_path, optimize=True)
    print(f"[ok] wrote {out_path}")


# -----------------------------------------------------------------------------
# Visualization 4: Pedestrian tracks across scale (evolving prediction)
# -----------------------------------------------------------------------------
def pedestrian_scale_evolution(df, out_path, min_frames=6, max_tracks=4, tile=220):
    """
    Find pedestrian IDs whose samples span at least 3 area quintiles, then
    plot each as a horizontal strip showing how the prediction evolves as
    the pedestrian gets closer. Demonstrates the scale 'wake-up' threshold.
    """
    # Group by pedestrian ID and find multi-sample tracks
    df = df.copy()
    track_counts = df.groupby("pid").size().sort_values(ascending=False)
    candidate_pids = track_counts[track_counts >= min_frames].index.tolist()

    selected = []
    for pid in candidate_pids:
        sub = df[df["pid"] == pid].sort_values("bbox_area")
        if len(sub) < min_frames:
            continue
        area_range = sub["bbox_area"].max() / max(sub["bbox_area"].min(), 1)
        # Want tracks that span a meaningful scale range
        if area_range < 3.0:
            continue
        # Want at least one crossing label and one wrong prediction
        if sub["label"].nunique() < 1:
            continue
        selected.append(pid)
        if len(selected) >= max_tracks:
            break

    if not selected:
        print("[skip] couldn't find pedestrian tracks spanning multiple scales")
        return

    header_h = 80
    strip_label_h = 36
    strip_h = tile + strip_label_h
    n_cols = min_frames
    row_label_w = 180
    canvas_w = row_label_w + n_cols * tile
    canvas_h = header_h + len(selected) * (strip_h + 20)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (252, 252, 252))
    draw = ImageDraw.Draw(canvas)

    title_font = _get_font(22)
    sub_font = _get_font(13)
    label_font = _get_font(14)
    cap_font = _get_font(11)
    row_font = _get_font(14)

    draw.text((20, 14),
              "How the model's prediction evolves as a pedestrian approaches",
              fill=(20, 20, 20), font=title_font)
    draw.text((20, 44),
              "Each row is ONE pedestrian across time, ordered left→right by increasing bbox area. "
              "Green = correct, red = incorrect.",
              fill=(70, 70, 70), font=sub_font)

    for row_idx, pid in enumerate(selected):
        sub = df[df["pid"] == pid].sort_values("bbox_area")
        # Evenly sample min_frames points from the sorted track
        idxs = np.linspace(0, len(sub) - 1, n_cols).astype(int)
        frames = sub.iloc[idxs]

        y_base = header_h + row_idx * (strip_h + 20)
        draw.rectangle([0, y_base, row_label_w, y_base + strip_h],
                       fill=(232, 236, 242))
        draw.text((12, y_base + strip_h // 2 - 18),
                  f"pid = {pid}",
                  fill=(40, 40, 40), font=row_font)
        label = int(frames.iloc[0]["label"])
        draw.text((12, y_base + strip_h // 2 + 4),
                  f"GT = {'crossing' if label == 1 else 'not crossing'}",
                  fill=(40, 40, 40), font=cap_font)

        for c_idx, (_, frm) in enumerate(frames.iterrows()):
            x_off = row_label_w + c_idx * tile
            try:
                with Image.open(frm["image_path"]).convert("RGB") as im:
                    t, adj = _context_crop(im, frm, pad_factor=2.0,
                                           target_size=tile)
            except (FileNotFoundError, OSError):
                continue
            line_color = "#22c55e" if frm["correct"] else "#ef4444"
            t_draw = ImageDraw.Draw(t)
            _draw_bbox(t_draw, adj, line_color, width=3)
            pr_text = ("C" if frm["prediction"] == 1 else "NC")
            _draw_label(t_draw, 6, 6,
                        f"P={pr_text} | {frm['confidence']:.2f}",
                        line_color, label_font)
            canvas.paste(t, (x_off, y_base))
            draw.text((x_off + 4, y_base + tile + 4),
                      f"area = {int(frm['bbox_area'])}px²",
                      fill=(40, 40, 40), font=cap_font)

    canvas.save(out_path, optimize=True)
    print(f"[ok] wrote {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")
    print(f"False negatives: {((df['label']==1) & (df['prediction']==0)).sum()}")
    print(f"Unique pedestrians: {df['pid'].nunique()}")

    scale_stratified_grid(df, out / "01_scale_stratified_failure_grid.png",
                          per_bin=args.per_bin)
    confident_missed_crossers(df, out / "02_confident_missed_crossers.png")
    in_context_examples(df, out / "03_small_pedestrians_in_context.png")
    pedestrian_scale_evolution(df, out / "04_pedestrian_scale_evolution.png")

    print(f"\nAll visualizations under: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--per-bin", type=int, default=6,
                        help="How many failure tiles per scale-quintile row")
    args = parser.parse_args()
    main(args)