import cv2
import numpy as np
import time

# =========================
# Black-tape rectangle detector + inside-STD (stable)
# Works well when the rectangle border is made of black tape and the inside is not black.
# Press 'q' to quit.
# =========================

# ---------- TUNING (start here) ----------
CAM_INDEX = 0

# Segment "black tape" using brightness (HSV V channel)
V_MAX = 85                 # ↑ if tape not detected, ↓ if too much background becomes black (try 60~120)

# Morphology to clean/connect the tape mask
KERNEL_OPEN = (3, 3)        # remove speckles
KERNEL_CLOSE = (11, 11)     # connect broken tape edges (try 7~15)
CLOSE_ITERS = 2             # increase if tape border breaks (1~3)

# Geometry filters
MIN_AREA = 3000             # ↑ ignore small junk, ↓ if your tape rectangle is small/far
EPS_RATIO = 0.02            # polygon approximation tolerance (0.015~0.04)
ASPECT_MIN, ASPECT_MAX = 0.35, 2.8

# Warp/ROI for inside statistics
WARP_W, WARP_H = 320, 240   # bigger = smoother std, but slower
INNER_MARGIN_FRAC = 0.10    # ignore tape border: fraction of min(w,h). Increase if tape is thick (0.06~0.18)

# Output behavior
PRINT_COOLDOWN_SEC = 1.0
SHOW_DEBUG_WINDOWS = True   # show mask/warp/roi windows
# ----------------------------------------


def order_points(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_quad(img, quad, out_w=WARP_W, out_h=WARP_H):
    """Perspective-warp a quad region into a fixed rectangle."""
    quad = order_points(quad)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


def detect_best_tape_quad(frame_bgr):
    """
    Detect the most reliable rectangle formed by black tape.
    Strategy:
      1) threshold low brightness (HSV V)
      2) open/close morphology
      3) find contours and approximate quads
      4) pick the best quad by area + border support
    Returns: best_quad (4x2 float), tape_mask (uint8), score (float)
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    # Tape mask: "dark pixels"
    tape_mask = cv2.inRange(v, 0, V_MAX)

    # Morphological cleanup
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_OPEN)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_CLOSE)
    tape_mask = cv2.morphologyEx(tape_mask, cv2.MORPH_OPEN, k_open, iterations=1)
    tape_mask = cv2.morphologyEx(tape_mask, cv2.MORPH_CLOSE, k_close, iterations=CLOSE_ITERS)

    contours, _ = cv2.findContours(tape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_score = -1e18

    # Use edges as an additional "reliability" signal (border support)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, EPS_RATIO * peri, True)

        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        quad = approx.reshape(4, 2).astype(np.float32)

        # Aspect filter (avoid extremely thin shapes)
        x, y, w, h = cv2.boundingRect(approx)
        if h <= 0 or w <= 0:
            continue
        aspect = w / float(h)
        if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
            continue

        # Border support: how many edge pixels lie near the quad border
        border_band = np.zeros(gray.shape, dtype=np.uint8)
        cv2.polylines(border_band, [approx], True, 255, 5)  # 5px band
        edge_support = cv2.mean(edges, mask=border_band)[0] / 255.0  # 0..1

        # Score: prefer bigger rectangles + strong border edges
        rect_area = w * h
        score = (np.sqrt(rect_area) * 0.8) + (edge_support * 250.0)

        if score > best_score:
            best_score = score
            best_quad = quad

    return best_quad, tape_mask, best_score


def compute_std_inside(frame_bgr, quad):
    """
    Warp the detected quad, then compute std on the interior (excluding tape border).
    Returns: std(float), warped_gray(uint8), roi_gray(uint8)
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    warped = warp_quad(gray, quad)

    h, w = warped.shape[:2]
    m = int(min(h, w) * INNER_MARGIN_FRAC)
    m = max(m, 2)

    roi = warped[m:h - m, m:w - m]
    if roi.size == 0:
        return None, warped, None

    std = float(np.std(roi))
    return std, warped, roi


def draw_quad(img, quad, color=(0, 255, 0), thickness=2):
    quad_int = order_points(quad).astype(int).reshape(-1, 1, 2)
    cv2.polylines(img, [quad_int], True, color, thickness)


def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {CAM_INDEX}")

    last_print = 0.0
    pTime = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()

        quad, tape_mask, score = detect_best_tape_quad(frame)

        if quad is not None:
            # Draw detected tape rectangle
            draw_quad(vis, quad, (0, 255, 0), 2)

            # Compute STD inside the rectangle (excluding tape border)
            std, warped, roi = compute_std_inside(frame, quad)

            if std is not None:
                cv2.putText(vis, f"std_inside={std:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                now = time.time()
                if now - last_print >= PRINT_COOLDOWN_SEC:
                    print(f"detected | std_inside={std:.2f} | score={score:.1f}")
                    last_print = now

                if SHOW_DEBUG_WINDOWS:
                    #cv2.imshow("Warped (gray)", warped)
                    #cv2.imshow("Warped ROI (inside)", roi)
                    a=1

        # FPS display
        cTime = time.time()
        fps = 1.0 / max(1e-6, (cTime - pTime))
        pTime = cTime
        cv2.putText(vis, f"{int(fps)} FPS", (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

        cv2.imshow("Webcam (detection)", vis)
        if SHOW_DEBUG_WINDOWS:
            cv2.imshow("Tape mask (V threshold)", tape_mask)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()