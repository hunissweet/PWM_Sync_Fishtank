"""
Module for detecting black square margins and analyzing intensity variation.
"""

import cv2
import numpy as np


class SquareDetector:
    """
    Detector for black square margins with intensity variation analysis.
    """
    
    def __init__(self, 
                 threshold_value=50,
                 min_area=1000,
                 max_area=500000,
                 epsilon_factor=0.02,
                 blur_kernel_size=5):
        """
        Initialize the square detector with tunable parameters.
        
        Args:
            threshold_value: Threshold for binary conversion (0-255)
            min_area: Minimum contour area to consider
            max_area: Maximum contour area to consider
            epsilon_factor: Approximation accuracy factor for contour (0.01-0.05)
            blur_kernel_size: Gaussian blur kernel size (odd number)
        """
        self.threshold_value = threshold_value
        self.min_area = min_area
        self.max_area = max_area
        self.epsilon_factor = epsilon_factor
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        
    def detect_square(self, frame):
        """
        Detect black square margin in the frame and analyze intensity variation.
        
        Args:
            frame: Input BGR image from webcam
            
        Returns:
            tuple: (detected: bool, annotated_frame: np.array, std_value: float)
                - detected: Whether a square was detected
                - annotated_frame: Frame with detection drawing
                - std_value: Standard deviation of intensity inside the box (0 if not detected)
        """
        # Create a copy for annotation
        annotated_frame = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # Apply binary threshold (inverse for black squares)
        _, binary = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = False
        std_value = 0.0
        best_square = None
        best_area = 0
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            # Approximate the contour to a polygon
            epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a quadrilateral
            if len(approx) == 4:
                # Check if it's roughly square-shaped
                if self._is_square_shaped(approx):
                    if area > best_area:
                        best_area = area
                        best_square = approx
        
        # Process the best detected square
        if best_square is not None:
            detected = True
            
            # Draw the detected square
            cv2.drawContours(annotated_frame, [best_square], -1, (0, 255, 0), 3)
            
            # Create a mask for the interior
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [best_square], -1, 255, -1)
            
            # Calculate standard deviation inside the box
            interior_pixels = gray[mask == 255]
            if len(interior_pixels) > 0:
                std_value = float(np.std(interior_pixels))
            
            # Add text annotations
            # Get bounding box for text placement
            x, y, w, h = cv2.boundingRect(best_square)
            
            # Draw detection status
            cv2.putText(annotated_frame, "DETECTED", (x, y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw standard deviation
            cv2.putText(annotated_frame, f"STD: {std_value:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw corner points
            for point in best_square:
                pt = tuple(point[0])
                cv2.circle(annotated_frame, pt, 5, (0, 0, 255), -1)
        else:
            # No square detected
            cv2.putText(annotated_frame, "NO SQUARE DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return detected, annotated_frame, std_value
    
    def _is_square_shaped(self, approx, aspect_ratio_tolerance=0.3):
        """
        Check if the approximated contour is roughly square-shaped.
        
        Args:
            approx: Approximated contour points
            aspect_ratio_tolerance: Tolerance for aspect ratio deviation from 1.0
            
        Returns:
            bool: True if shape is square-like
        """
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)
        
        # Calculate aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Check if aspect ratio is close to 1 (square)
        if abs(aspect_ratio - 1.0) > aspect_ratio_tolerance:
            return False
        
        # Additional check: area ratio between contour and bounding box
        contour_area = cv2.contourArea(approx)
        bbox_area = w * h
        
        if bbox_area > 0:
            area_ratio = contour_area / bbox_area
            # Square should fill most of its bounding box (>0.7)
            return area_ratio > 0.7
        
        return False
    
    def update_parameters(self, **kwargs):
        """
        Update detection parameters dynamically.
        
        Accepted parameters:
            - threshold_value
            - min_area
            - max_area
            - epsilon_factor
            - blur_kernel_size
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key == 'blur_kernel_size':
                    # Ensure kernel size is odd
                    value = value if value % 2 == 1 else value + 1
                setattr(self, key, value)
                print(f"Updated {key} to {value}")
            else:
                print(f"Warning: {key} is not a valid parameter")