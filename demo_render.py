"""Rendering for the 4-view demo display.

Every view is a QLabel that shows a BGR image, so everything a view needs to say --
frames, top-5 bars, streaming text, "load failed" -- is drawn into an image here and
handed to the same setPixmap path. That keeps one rendering contract for all model
kinds and means a view can always show *something*: a blank view reads as a crash.

The device badge is the point of the demo (which model landed on which device), so
it is drawn on every frame, in a fixed colour per device.
"""
import cv2
import numpy as np

VIEW_W, VIEW_H = 640, 480

# BGR. Distinct hues so the placement is readable across the 2x2 grid at a glance.
DEVICE_COLORS = {
    "cpu": (219, 152, 52),    # blue
    "npu": (94, 185, 90),     # green
    "gpu": (60, 145, 245),    # orange
}
_UNKNOWN_COLOR = (128, 128, 128)
_HEADER_H = 34
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def device_color(device):
    return DEVICE_COLORS.get(str(device or "").lower(), _UNKNOWN_COLOR)


def draw_header(img, model, device, metric="", error=None):
    """Model name + coloured device badge + live metric, drawn across the top.

    Mutates and returns `img`. Never raises: a header that throws would take down
    the view it is supposed to be labelling.
    """
    try:
        h, w = img.shape[:2]
        dev = str(device or "?").upper()
        color = device_color(device)

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, _HEADER_H), (32, 32, 32), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        cv2.putText(img, str(model or "-"), (8, 24), _FONT, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

        (tw, _), _ = cv2.getTextSize(str(model or "-"), _FONT, 0.62, 2)
        bx = 8 + tw + 12
        (bw, _), _ = cv2.getTextSize(dev, _FONT, 0.55, 2)
        cv2.rectangle(img, (bx, 6), (bx + bw + 16, _HEADER_H - 6), color, -1)
        cv2.putText(img, dev, (bx + 8, 24), _FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        if metric:
            (mw, _), _ = cv2.getTextSize(metric, _FONT, 0.55, 2)
            cv2.putText(img, metric, (w - mw - 8, 24), _FONT, 0.55, (230, 230, 230), 2, cv2.LINE_AA)
        if error:
            cv2.rectangle(img, (0, _HEADER_H), (w, _HEADER_H + 26), (40, 40, 190), -1)
            cv2.putText(img, str(error)[:60], (8, _HEADER_H + 19), _FONT, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
    except Exception:
        pass
    return img


def placeholder(model, device, message, w=VIEW_W, h=VIEW_H):
    """A view that cannot run still shows its model, its device and why."""
    img = np.full((h, w, 3), 24, dtype=np.uint8)
    cv2.putText(img, str(message), (16, h // 2), _FONT, 0.7, (90, 90, 235), 2, cv2.LINE_AA)
    return draw_header(img, model, device, "")


def _wrap(text, width_px, scale, thickness):
    """Greedy wrap on character width; good enough for a monospace-ish demo card."""
    lines, cur = [], ""
    for word in str(text).split():
        trial = (cur + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(trial, _FONT, scale, thickness)
        if tw <= width_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def text_card(text, model, device, metric="", image=None, generating=True,
              w=VIEW_W, h=VIEW_H, prompt=None):
    """Generated text, optionally beside the image that prompted it (VLM).

    `generating` draws a block cursor so a slow generation still looks alive rather
    than frozen -- the whole point of streaming it token by token.
    """
    img = np.full((h, w, 3), 24, dtype=np.uint8)
    text_x0 = 8
    if image is not None:
        try:
            half = w // 2
            thumb_h = h - _HEADER_H - 12
            thumb = cv2.resize(image, (half - 12, thumb_h))
            img[_HEADER_H + 6:_HEADER_H + 6 + thumb_h, 6:6 + (half - 12)] = thumb
            text_x0 = half + 4
        except Exception:
            text_x0 = 8

    avail = w - text_x0 - 8
    y = _HEADER_H + 26
    if prompt:
        for line in _wrap(f"> {prompt}", avail, 0.44, 1)[:2]:
            cv2.putText(img, line, (text_x0, y), _FONT, 0.44, (140, 190, 140), 1, cv2.LINE_AA)
            y += 18
        y += 6

    body = _wrap(text or "", avail, 0.5, 1)
    max_lines = max(1, (h - y - 10) // 20)
    # Keep the tail: a demo should show what is being generated now, not the start
    # of an answer that has already scrolled past.
    for line in body[-max_lines:]:
        cv2.putText(img, line, (text_x0, y), _FONT, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        y += 20
    if generating:
        last = body[-1] if body else ""
        (lw, _), _ = cv2.getTextSize(last, _FONT, 0.5, 1)
        cx = text_x0 + (lw if body else 0) + 3
        cv2.rectangle(img, (cx, y - 34 if body else y - 14), (cx + 8, y - 20 if body else y),
                      (235, 235, 235), -1)
    return draw_header(img, model, device, metric)


def top5_panel(frame, top5, model, device, metric="", w=VIEW_W, h=VIEW_H):
    """Classified image with a top-5 probability bar chart under it."""
    img = np.full((h, w, 3), 24, dtype=np.uint8)
    chart_h = 118
    try:
        pic_h = h - chart_h - _HEADER_H
        pic = cv2.resize(frame, (w, pic_h))
        img[_HEADER_H:_HEADER_H + pic_h] = pic
    except Exception:
        pass

    y = h - chart_h + 12
    for cid_name, prob in (top5 or [])[:5]:
        label = str(cid_name)[:28]
        p = max(0.0, min(1.0, float(prob)))
        cv2.putText(img, label, (8, y + 9), _FONT, 0.42, (225, 225, 225), 1, cv2.LINE_AA)
        bar_x0 = 210
        bar_w = int((w - bar_x0 - 52) * p)
        cv2.rectangle(img, (bar_x0, y), (w - 52, y + 12), (60, 60, 60), -1)
        if bar_w > 0:
            cv2.rectangle(img, (bar_x0, y), (bar_x0 + bar_w, y + 12), device_color(device), -1)
        cv2.putText(img, f"{p * 100:4.1f}%", (w - 48, y + 10), _FONT, 0.4,
                    (225, 225, 225), 1, cv2.LINE_AA)
        y += 21
    return draw_header(img, model, device, metric)
