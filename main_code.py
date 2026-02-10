"""
Main script for webcam-based black square detection with intensity analysis.
"""

import cv2
import sys
from square_detector import SquareDetector


def create_trackbars(window_name, detector):
    """Create trackbars for tuning detection parameters in real-time."""
    
    def nothing(x):
        pass
    
    # Create trackbars
    cv2.createTrackbar('Threshold', window_name, detector.threshold_value, 255, nothing)
    cv2.createTrackbar('Min Area', window_name, detector.min_area // 100, 5000, nothing)
    cv2.createTrackbar('Max Area', window_name, detector.max_area // 1000, 1000, nothing)
    cv2.createTrackbar('Epsilon x100', window_name, int(detector.epsilon_factor * 100), 10, nothing)
    cv2.createTrackbar('Blur Kernel', window_name, detector.blur_kernel_size, 31, nothing)


def update_parameters_from_trackbars(window_name, detector):
    """Update detector parameters from trackbar values."""
    
    threshold = cv2.getTrackbarPos('Threshold', window_name)
    min_area = cv2.getTrackbarPos('Min Area', window_name) * 100
    max_area = cv2.getTrackbarPos('Max Area', window_name) * 1000
    epsilon = cv2.getTrackbarPos('Epsilon x100', window_name) / 100.0
    blur = cv2.getTrackbarPos('Blur Kernel', window_name)
    
    # Ensure minimum values
    min_area = max(100, min_area)
    max_area = max(1000, max_area)
    blur = max(1, blur)
    epsilon = max(0.01, epsilon)
    
    detector.update_parameters(
        threshold_value=threshold,
        min_area=min_area,
        max_area=max_area,
        epsilon_factor=epsilon,
        blur_kernel_size=blur
    )


def main():
    """Main function to run the webcam detection loop."""
    
    # Initialize the square detector with default parameters
    detector = SquareDetector(
        threshold_value=50,
        min_area=1000,
        max_area=500000,
        epsilon_factor=0.02,
        blur_kernel_size=5
    )
    
    # Initialize webcam (0 is default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create windows
    window_name = 'Square Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Create trackbars for parameter tuning
    create_trackbars(window_name, detector)
    
    print("=" * 60)
    print("Black Square Margin Detection")
    print("=" * 60)
    print("Controls:")
    print("  - Use trackbars to tune detection parameters")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset parameters to defaults")
    print("=" * 60)
    
    frame_count = 0
    
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Update parameters from trackbars every 5 frames to reduce overhead
        if frame_count % 5 == 0:
            update_parameters_from_trackbars(window_name, detector)
        
        # Detect square and get results
        detected, annotated_frame, std_value = detector.detect_square(frame)
        
        # Add additional information overlay
        info_text = f"Frame: {frame_count} | Detection: {'YES' if detected else 'NO'}"
        cv2.putText(annotated_frame, info_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the result
        cv2.imshow(window_name, annotated_frame)
        
        # Print detection status to console (optional, every 30 frames)
        if frame_count % 30 == 0:
            status = "DETECTED" if detected else "NOT DETECTED"
            print(f"Frame {frame_count}: {status} | STD: {std_value:.2f}")
        
        frame_count += 1
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('r'):
            print("\nResetting parameters to defaults...")
            detector = SquareDetector(
                threshold_value=50,
                min_area=1000,
                max_area=500000,
                epsilon_factor=0.02,
                blur_kernel_size=5
            )
            # Update trackbars
            cv2.setTrackbarPos('Threshold', window_name, 50)
            cv2.setTrackbarPos('Min Area', window_name, 10)
            cv2.setTrackbarPos('Max Area', window_name, 500)
            cv2.setTrackbarPos('Epsilon x100', window_name, 2)
            cv2.setTrackbarPos('Blur Kernel', window_name, 5)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nCleaned up resources. Goodbye!")


if __name__ == "__main__":
    main()