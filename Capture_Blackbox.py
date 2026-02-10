import cv2
import numpy as np
import time

# ---------- TUNING ----------
MIN_AREA = 100
CANNY1, CANNY2 = 50, 180
CLOSE_KERNEL = (4, 4)
EPS_RATIO = 0.02
ASPECT_MIN, ASPECT_MAX = 0.35, 2.8
FILL_RATIO_MIN = 0.20      # border-only rectangles often need low fill ratio
PRINT_COOLDOWN_SEC = 1.0

# ROI warp size for intensity stats (bigger = more stable, slower)
WARP_W, WARP_H = 220, 160
# ---------------------------

def order_points(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def quad_score(gray, edges, quad):
    """
    Reliability score for a quad. Higher = better.
    Uses: area, fill ratio, edge support near border.
    """
    # Bounding box area and aspect
    x, y, w, h = cv2.boundingRect(quad)
    if w <= 0 or h <= 0:
        return -1e9

    rect_area = w * h
    cnt_area = cv2.contourArea(quad)  # note: approx quad area, not original contour
    aspect = w / float(h)

    if rect_area <= 0:
        return -1e9

    fill_ratio = cnt_area / float(rect_area) if rect_area else 0.0
    if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
        return -1e9
    if fill_ratio < FILL_RATIO_MIN:
        return -1e9

    # Edge support: how many edge pixels lie on/near the quad border
    border_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.polylines(border_mask, [quad], True, 255, 5)  # thickness=5 pixels band
    edge_support = cv2.mean(edges, mask=border_mask)[0] / 255.0  # 0..1

    # Favor larger rectangles with strong border edges
    # Use sqrt(area) to avoid giant domination
    score = (np.sqrt(rect_area) * 0.6) + (edge_support * 200.0) + (fill_ratio * 50.0)
    return score

def warp_quad(gray, quad, out_w=WARP_W, out_h=WARP_H):
    pts = order_points(quad)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(gray, M, (out_w, out_h))
    return warped

def detect_best_rectangle(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray_blur, CANNY1, CANNY2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_KERNEL)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_score = -1e18

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

        quad = approx  # 4x1x2
        score = quad_score(gray, edges, quad)
        if score > best_score:
            best_score = score
            best_quad = quad

    return best_quad, gray, edges, edges_closed

def intensity_variation_inside(gray, quad):
    """
    Returns stats for intensity variation inside the detected rectangle.
    Uses a perspective warp so you measure a normalized ROI.
    """
    warped = warp_quad(gray, quad)

    # Optionally ignore a small border margin so you measure the interior (not the black outline)
    m = 8
    if warped.shape[0] > 2*m and warped.shape[1] > 2*m:
        roi = warped[m:-m, m:-m]
    else:
        roi = warped

    
    std = float(np.std(roi))
    

    return std

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    cTime = 0
    pTime = 0
    last_print = 0.0


    while True:
        ok, frame = cap.read()
        if not ok:
            break

        #frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        

        best_quad, gray, edges, edges_closed = detect_best_rectangle(frame)

        vis = frame.copy()

        if best_quad is not None:
            # Draw best quad
            cv2.polylines(vis, [best_quad], True, (0, 255, 0), 2)

            # Compute intensity variation inside
            std = intensity_variation_inside(gray, best_quad)

            # Overlay info on image
            txt = f"std={std:.1f}"
            cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Print with cooldown
            now = time.time()
            if now - last_print >= PRINT_COOLDOWN_SEC:
                print(f"detected  std={std:.2f}")
                last_print = now

            # Show warped rectangle views (useful for debugging)
            # cv2.imshow("Warped rect", warped)
            # cv2.imshow("Warped interior ROI", roi)
        else:
            # If none detected, close the extra windows if they were opened before
            # (optional; leaving them is okay too)
            pass
        cTime = time.time()
        fps=1/(cTime-pTime)
        pTime = cTime
        cv2.putText(vis,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
        cv2.imshow("Webcam (best detection)", vis)
        cv2.imshow("Edges", edges)
        cv2.imshow("Edges (closed)", edges_closed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
