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

# Additional parameters for better detection
GAUSSIAN_BLUR = (5, 5)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
MORPH_ITERATIONS = 2

# Adaptive threshold parameters
USE_ADAPTIVE = False  # Set to True if lighting is uneven
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2
# ---------------------------


def create_trackbars():
    """Create tuning trackbars for real-time parameter adjustment."""
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 500, 400)
    
    cv2.createTrackbar('Canny Low', 'Controls', CANNY1, 255, lambda x: None)
    cv2.createTrackbar('Canny High', 'Controls', CANNY2, 255, lambda x: None)
    cv2.createTrackbar('Close Kernel', 'Controls', CLOSE_KERNEL[0], 20, lambda x: None)
    cv2.createTrackbar('Morph Iter', 'Controls', MORPH_ITERATIONS, 10, lambda x: None)
    cv2.createTrackbar('EPS x100', 'Controls', int(EPS_RATIO * 100), 10, lambda x: None)
    cv2.createTrackbar('Min Area', 'Controls', MIN_AREA // 10, 500, lambda x: None)
    cv2.createTrackbar('Bilateral D', 'Controls', BILATERAL_D, 15, lambda x: None)
    cv2.createTrackbar('Use Adaptive', 'Controls', 0, 1, lambda x: None)


def get_trackbar_values():
    """Get current trackbar values."""
    params = {}
    params['canny1'] = max(1, cv2.getTrackbarPos('Canny Low', 'Controls'))
    params['canny2'] = max(1, cv2.getTrackbarPos('Canny High', 'Controls'))
    params['close_kernel'] = max(1, cv2.getTrackbarPos('Close Kernel', 'Controls'))
    params['morph_iter'] = max(1, cv2.getTrackbarPos('Morph Iter', 'Controls'))
    params['eps_ratio'] = max(0.01, cv2.getTrackbarPos('EPS x100', 'Controls') / 100.0)
    params['min_area'] = max(10, cv2.getTrackbarPos('Min Area', 'Controls') * 10)
    params['bilateral_d'] = max(3, cv2.getTrackbarPos('Bilateral D', 'Controls'))
    if params['bilateral_d'] % 2 == 0:
        params['bilateral_d'] += 1
    params['use_adaptive'] = cv2.getTrackbarPos('Use Adaptive', 'Controls') == 1
    return params


def order_points(pts):
    """Order points in clockwise order: TL, TR, BR, BL."""
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
    cnt_area = cv2.contourArea(quad)
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
    cv2.polylines(border_mask, [quad], True, 255, 5)
    edge_support = cv2.mean(edges, mask=border_mask)[0] / 255.0

    # Corner detection score - good rectangles have strong corners
    corners = quad.reshape(4, 2)
    corner_score = 0
    for corner in corners:
        x, y = int(corner[0]), int(corner[1])
        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            # Check gradient strength at corners
            if x > 0 and y > 0 and x < gray.shape[1]-1 and y < gray.shape[0]-1:
                gx = float(gray[y, x+1]) - float(gray[y, x-1])
                gy = float(gray[y+1, x]) - float(gray[y-1, x])
                gradient_mag = np.sqrt(gx*gx + gy*gy)
                corner_score += gradient_mag

    corner_score /= 4.0  # Average

    # Favor larger rectangles with strong border edges and corners
    score = (np.sqrt(rect_area) * 0.5) + (edge_support * 150.0) + (fill_ratio * 30.0) + (corner_score * 0.3)
    return score


def warp_quad(gray, quad, out_w=WARP_W, out_h=WARP_H):
    """Warp quadrilateral to rectangular view."""
    pts = order_points(quad)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(gray, M, (out_w, out_h))
    return warped


def preprocess_frame(gray, params):
    """
    Enhanced preprocessing with multiple techniques.
    Returns preprocessed grayscale and edge map.
    """
    # Apply bilateral filter to reduce noise while preserving edges
    if params['bilateral_d'] > 0:
        filtered = cv2.bilateralFilter(gray, params['bilateral_d'], 
                                       BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    else:
        filtered = gray.copy()
    
    # Additional Gaussian blur for edge detection
    gray_blur = cv2.GaussianBlur(filtered, GAUSSIAN_BLUR, 0)
    
    # Adaptive threshold for uneven lighting (optional)
    if params['use_adaptive']:
        adaptive = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
        # Combine with Canny
        edges_adaptive = cv2.Canny(adaptive, params['canny1'], params['canny2'])
        edges_normal = cv2.Canny(gray_blur, params['canny1'], params['canny2'])
        edges = cv2.bitwise_or(edges_adaptive, edges_normal)
    else:
        edges = cv2.Canny(gray_blur, params['canny1'], params['canny2'])
    
    # Morphological closing to connect edge segments
    kernel_size = max(1, params['close_kernel'])
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, 
                                    iterations=params['morph_iter'])
    
    # Optional: dilate slightly to make edges thicker for better contour detection
    edges_dilated = cv2.dilate(edges_closed, kernel, iterations=1)
    
    return filtered, edges, edges_dilated


def detect_best_rectangle(frame_bgr, params):
    """
    Detect the best rectangle in the frame using edge detection and scoring.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Preprocess with enhanced techniques
    filtered, edges_raw, edges_processed = preprocess_frame(gray, params)
    
    # Find contours
    contours, _ = cv2.findContours(edges_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_quad = None
    best_score = -1e18
    all_quads = []  # Store all valid quads for debugging

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params['min_area']:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, params['eps_ratio'] * peri, True)

        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        quad = approx
        score = quad_score(gray, edges_raw, quad)
        
        if score > -1e9:  # Valid quad
            all_quads.append((quad, score))
        
        if score > best_score:
            best_score = score
            best_quad = quad

    return best_quad, gray, edges_raw, edges_processed, all_quads


def intensity_variation_inside(gray, quad):
    """
    Returns stats for intensity variation inside the detected rectangle.
    Uses a perspective warp so you measure a normalized ROI.
    """
    warped = warp_quad(gray, quad)

    # Ignore border margin to measure interior only (not the black outline)
    m = 8
    if warped.shape[0] > 2*m and warped.shape[1] > 2*m:
        roi = warped[m:-m, m:-m]
    else:
        roi = warped

    std = float(np.std(roi))
    mean = float(np.mean(roi))
    
    return std, mean, warped, roi


def draw_debug_info(vis, best_quad, std, mean, fps, all_quads):
    """Draw comprehensive debug information on the visualization."""
    if best_quad is not None:
        # Draw best quad in green with thick line
        cv2.polylines(vis, [best_quad], True, (0, 255, 0), 3)
        
        # Draw corner points
        for pt in best_quad.reshape(4, 2):
            cv2.circle(vis, tuple(pt.astype(int)), 6, (0, 0, 255), -1)
        
        # Draw other detected quads in yellow (for debugging)
        for quad, score in all_quads[:5]:  # Show top 5
            if not np.array_equal(quad, best_quad):
                cv2.polylines(vis, [quad], True, (0, 255, 255), 1)
        
        # Info overlay with background for readability
        info_y = 25
        cv2.rectangle(vis, (5, 5), (350, 95), (0, 0, 0), -1)
        cv2.rectangle(vis, (5, 5), (350, 95), (0, 255, 0), 2)
        
        cv2.putText(vis, f"DETECTED", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"STD: {std:.2f}", (10, info_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Mean: {mean:.2f}", (10, info_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        # No detection
        cv2.rectangle(vis, (5, 5), (250, 35), (0, 0, 0), -1)
        cv2.rectangle(vis, (5, 5), (250, 35), (0, 0, 255), 2)
        cv2.putText(vis, "NO DETECTION", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # FPS counter
    cv2.putText(vis, f"FPS: {int(fps)}", (10, vis.shape[0] - 15), 
               cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Create control panel
    create_trackbars()
    
    # Initialize parameters
    params = {
        'canny1': CANNY1,
        'canny2': CANNY2,
        'close_kernel': CLOSE_KERNEL[0],
        'morph_iter': MORPH_ITERATIONS,
        'eps_ratio': EPS_RATIO,
        'min_area': MIN_AREA,
        'bilateral_d': BILATERAL_D,
        'use_adaptive': USE_ADAPTIVE
    }
    
    pTime = 0
    last_print = 0.0
    
    print("=" * 60)
    print("Enhanced Black Square Detection")
    print("=" * 60)
    print("Controls:")
    print("  - Adjust trackbars in 'Controls' window for real-time tuning")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press 'r' to reset trackbars")
    print("=" * 60)

    frame_count = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Update parameters from trackbars every 5 frames
        if frame_count % 5 == 0:
            params = get_trackbar_values()

        # Detect rectangle
        best_quad, gray, edges_raw, edges_processed, all_quads = detect_best_rectangle(frame, params)

        # Create visualization
        vis = frame.copy()

        if best_quad is not None:
            # Compute intensity variation inside
            std, mean, warped, roi = intensity_variation_inside(gray, best_quad)

            # Calculate FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime) if pTime > 0 else 0
            pTime = cTime

            # Draw debug information
            draw_debug_info(vis, best_quad, std, mean, fps, all_quads)

            # Print with cooldown
            now = time.time()
            if now - last_print >= PRINT_COOLDOWN_SEC:
                print(f"Frame {frame_count}: DETECTED | STD={std:.2f} | Mean={mean:.2f} | FPS={int(fps)}")
                last_print = now

            # Show warped views for debugging
            cv2.imshow("Warped Rectangle", warped)
            cv2.imshow("Interior ROI", roi)
        else:
            cTime = time.time()
            fps = 1 / (cTime - pTime) if pTime > 0 else 0
            pTime = cTime
            draw_debug_info(vis, None, 0, 0, fps, [])

        # Display windows
        cv2.imshow("Detection", vis)
        cv2.imshow("Edges (Raw)", edges_raw)
        cv2.imshow("Edges (Processed)", edges_processed)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{frame_count}.png"
            cv2.imwrite(filename, vis)
            print(f"Saved frame as {filename}")
        elif key == ord('r'):
            # Reset trackbars to defaults
            cv2.setTrackbarPos('Canny Low', 'Controls', CANNY1)
            cv2.setTrackbarPos('Canny High', 'Controls', CANNY2)
            cv2.setTrackbarPos('Close Kernel', 'Controls', CLOSE_KERNEL[0])
            cv2.setTrackbarPos('Morph Iter', 'Controls', MORPH_ITERATIONS)
            cv2.setTrackbarPos('EPS x100', 'Controls', int(EPS_RATIO * 100))
            cv2.setTrackbarPos('Min Area', 'Controls', MIN_AREA // 10)
            cv2.setTrackbarPos('Bilateral D', 'Controls', BILATERAL_D)
            cv2.setTrackbarPos('Use Adaptive', 'Controls', 0)
            print("Reset trackbars to defaults")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\nSession complete!")


if __name__ == "__main__":
    main()