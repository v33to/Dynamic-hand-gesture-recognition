import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINETYPE = cv2.LINE_AA
FONTSCALE = 0.5
COLOR = (0, 255, 0)
THICKNESS = 2
OFFSET = 50

# Colors
GREEN = (0, 202, 105)
ORANGE = (0, 181, 249)
BLUE = (224, 175, 14)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)


def draw_landmarks(landmarks, frame):
    """Draw landmarks on a hand."""
    # Joint indexes.
    # Visit https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    # for more details
    #
    #        8   12  16  20
    #        |   |   |   |
    #        7   11  15  19
    #    4   |   |   |   |
    #    |   6   10  14  18
    #    3   |   |   |   |
    #    |   5---9---13--17
    #    2    \         /
    #     \    \       /
    #      1    \     /
    #       \    \   /
    #        ------0-
    #
    connections = [
        (1, 2),
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (7, 8),
        (9, 10),
        (10, 11),
        (11, 12),
        (13, 14),
        (14, 15),
        (15, 16),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 1),
        (0, 5),
        (0, 9),
        (0, 13),
        (0, 17),
        (5, 9),
        (9, 13),
        (13, 17),
    ]

    if landmarks is not None:
        for connection in connections:
            x_0, y_0 = landmarks[connection[0]][:2]
            x_1, y_1 = landmarks[connection[1]][:2]
            cv2.line(frame, (int(x_0), int(y_0)), (int(x_1), int(y_1)), BLUE, 2)

        for index, point in enumerate(landmarks):
            x, y = point[:2]
            if index in [0, 1, 2, 5, 9, 13, 17]:  # Palm
                cv2.circle(frame, (int(x), int(y)), 6, ORANGE, -1)
            else:
                cv2.circle(frame, (int(x), int(y)), 6, GREEN, -1)


def draw_handbbox(hand_bbox, frame):
    """Draw a bounding box enclosing a hand."""
    rot_degrees = hand_bbox.rotation * 180 / np.pi
    rect = (hand_bbox.center, hand_bbox.dims, rot_degrees)

    # Draw hand bbox
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    cv2.drawContours(frame, [box], 0, GREEN, 2)


def hide_hand(hand_bbox, frame):
    """Fill the bounding box to 'hide' a hand bbox."""
    rot_degrees = hand_bbox.rotation * 180 / np.pi
    rect = (hand_bbox.center, hand_bbox.dims, rot_degrees)

    # Draw hand bbox
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    cv2.drawContours(frame, [box], 0, GREEN, cv2.FILLED)


def draw_overlay_continuous(frame, status, is_processing, processing_elapsed, 
                            prediction_result, gesture_names, processing_display_duration):
    """
    Draw overlay information for continuous recognition mode
    
    Args:
        frame: Video frame to draw on
        status: Buffer status dictionary
        is_processing: Whether currently in processing/display mode
        processing_elapsed: Time elapsed since processing started
        prediction_result: Classification result dictionary
        gesture_names: Dictionary mapping class indices to gesture names
        processing_display_duration: How long to display results
    """
    h, w = frame.shape[:2]
    overlay_height = 80
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - overlay_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Progress bar
    bar_width = 250
    bar_height = 20
    bar_x = 20
    bar_y = h - overlay_height + 15
    
    # Draw background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    
    if is_processing:
        # Show processing countdown
        progress = min(1.0, processing_elapsed / processing_display_duration)
        fill_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), ORANGE, -1)
    else:
        # Show buffer fill progress
        progress = min(1.0, status['frame_count'] / status['max_frames'])
        fill_width = int(bar_width * progress)
        
        if status['is_ready']:
            color = GREEN  # Green when ready
        else:
            color = (100, 100, 100)  # Gray when collecting
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
    
    # Draw border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
    
    # Buffer status text
    buffer_text_y = bar_y + bar_height + 20
    buffer_text = f"Buffer: {status['frame_count']}/{status['max_frames']}"
    if status['null_count'] > 0:
        buffer_text += f" (Nulls: {status['null_count']})"
    cv2.putText(frame, buffer_text, (bar_x, buffer_text_y), 
               FONT, 0.5, (200, 200, 200), 1)
    
    # Prediction/status text
    text_x = bar_x + bar_width + 30
    text_y_line1 = h - overlay_height + 25
    text_y_line2 = h - overlay_height + 50
    text_y_line3 = h - overlay_height + 70
    
    if is_processing and prediction_result:
        remaining = max(0, processing_display_duration - processing_elapsed)
        
        if prediction_result['is_unknown']:
            # Display raw class for unknown predictions
            raw_label = gesture_names.get(prediction_result['predicted_class'], 
                                          f"Class {prediction_result['predicted_class']}")
            result_text = f"UNKNOWN ({raw_label})"
            cv2.putText(frame, result_text, (text_x, text_y_line1), 
                       FONT, 0.6, RED, 2)
            
            conf_text = f"C:{prediction_result['max_confidence']:.0%} | E:{prediction_result['entropy']:.2f}"
            cv2.putText(frame, conf_text, (text_x, text_y_line2), 
                       FONT, 0.45, (200, 200, 200), 1)
        else:
            label = gesture_names.get(prediction_result['predicted_class'], 
                                     f"Class {prediction_result['predicted_class']}")
            result_text = f"{label}"
            cv2.putText(frame, result_text, (text_x, text_y_line1), 
                       FONT, 0.7, GREEN, 2)
            
            conf_text = f"Confidence: {prediction_result['max_confidence']:.0%} | Entropy: {prediction_result['entropy']:.2f}"
            cv2.putText(frame, conf_text, (text_x, text_y_line2), 
                       FONT, 0.45, (200, 200, 200), 1)
        
        if prediction_result.get('voting_results'):
            voting_text = f"Windows: {prediction_result['num_valid']}/{prediction_result['num_windows']} valid"
            cv2.putText(frame, voting_text, (text_x, text_y_line3), 
                       FONT, 0.45, (200, 200, 200), 1)
            
    elif status['consecutive_nulls'] > 0:
        warning_text = f"Missing detection ({status['consecutive_nulls']}/2)"
        cv2.putText(frame, warning_text, (text_x, text_y_line1), 
                   FONT, 0.55, ORANGE, 2)
    elif status['is_ready']:
        ready_text = "READY - Collecting data..."
        cv2.putText(frame, ready_text, (text_x, text_y_line1), 
                   FONT, 0.6, GREEN, 2)
    else:
        collecting_text = f"Collecting... ({status['frame_count']}/{status['min_frames']} minimum)"
        cv2.putText(frame, collecting_text, (text_x, text_y_line1), 
                   FONT, 0.55, (200, 200, 200), 1)


def draw_fps(fps, frame):
    """Draw FPS with color coding based on performance."""
    fps_text = f"FPS: {fps:.1f}"
    
    # Color based on FPS value
    if fps < 10:
        color = RED  # Red for less than 10 FPS
    elif fps < 20:
        color = YELLOW  # Yellow for 10-20 FPS
    else:
        color = GREEN  # Green for 20+ FPS
    
    cv2.putText(frame, fps_text, (10, 30), FONT, 0.7, color, 2, LINETYPE)