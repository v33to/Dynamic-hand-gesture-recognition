import os
import sys
import time
import logging
import argparse

import cv2

import hand_tracker_seq
import gesture_classifier_seq
from app_utils import drawkit_seq

# GoPoint
if os.path.isdir("/opt/gopoint-apps/scripts/machine_learning/imx_gesture_recognition"):
    sys.path.append("/opt/gopoint-apps/scripts/machine_learning/imx_gesture_recognition")

# Continuous recognition parameters
MIN_FRAMES = 10  # Minimum frames before processing
MAX_FRAMES = 20  # Maximum frames to capture
SEQUENCE_LENGTH = 10  # Size of each sliding window
MAX_SLIDING_WINDOWS = 7  # Maximum number of sliding windows to analyze

PROCESSING_DISPLAY_DURATION = 4.0  # Duration to display results after processing

GESTURE_NAMES = {
    0: "Swipe Up",
    1: "Swipe Down",
    2: "Swipe Left",
    3: "Swipe Right",
    4: "Zoom In",
    5: "Zoom Out",
    6: "Rotate Clockwise",
    7: "Rotate Counter Clockwise",
    8: "Unknown"
}


def run(stream, args):
    """Run Continuous Sequential Hand Gesture Recognition task"""
    cap = cv2.VideoCapture(stream)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video stream or file: {stream}")

    tracker = hand_tracker_seq.HandTracker(
        palm_detection_model=args.palm_model,
        hand_landmark_model=args.hand_landmark_model,
        anchors=args.anchors,
        external_delegate=args.external_delegate_path,
        num_hands=args.num_hands,
    )

    classifier = gesture_classifier_seq.Classifier(
        model=args.classification_model,
        external_delegate=args.external_delegate_path
    )
    
    # Initialize sequence buffer for continuous recognition
    sequence_buffer = gesture_classifier_seq.SequenceBuffer(
        min_frames=MIN_FRAMES,
        max_frames=MAX_FRAMES,
        sequence_length=SEQUENCE_LENGTH,
        feature_size=63
    )
    
    cv2.namedWindow("i.MX Continuous Sequential Gesture Recognition", cv2.WND_PROP_FULLSCREEN)
    
    # Processing state
    is_processing = False
    processing_start_time = None
    last_prediction_result = None
    
    # FPS tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    logging.info("Starting continuous gesture recognition with sliding windows")
    logging.info(f"Min frames: {MIN_FRAMES}, Max frames: {MAX_FRAMES}")
    logging.info(f"Sequence length: {SEQUENCE_LENGTH}, Max windows: {MAX_SLIDING_WINDOWS}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        
        # Calculate FPS
        fps_frame_count += 1
        fps_elapsed = time.time() - fps_start_time
        if fps_elapsed >= 1.0:  # Update FPS every second
            current_fps = fps_frame_count / fps_elapsed
            fps_frame_count = 0
            fps_start_time = time.time()
        
        # Check if we're in processing/display mode
        processing_elapsed = 0
        if is_processing:
            processing_elapsed = time.time() - processing_start_time
            if processing_elapsed >= PROCESSING_DISPLAY_DURATION:
                # Processing display period is over, reset buffer and resume
                sequence_buffer.reset("Processing complete")
                is_processing = False
                processing_start_time = None
                last_prediction_result = None
                logging.debug("Processing complete, resuming capture")
        
        # Continue capturing frames if not processing
        if not is_processing:
            detections = tracker(frame)
            
            if detections:
                for results in detections:
                    landmarks, hand_bbox, handedness = results
                    drawkit_seq.draw_landmarks(landmarks, frame)
                    drawkit_seq.draw_handbbox(hand_bbox, frame)
                    
                    sequence_buffer.add_frame(landmarks)
            else:
                result = sequence_buffer.add_null_frame()
                
                # Check if hand was lost and we should process
                if result == 'process':
                    logging.debug("Hand lost, triggering processing")
                    # Hand lost after capturing sufficient frames
                    # Get sliding windows (this will automatically clean null frames)
                    windows = sequence_buffer.get_sliding_windows(MAX_SLIDING_WINDOWS)
                    
                    if windows:
                        logging.info(f"Processing {len(windows)} sliding windows")
                        # Perform classification with majority voting
                        prediction_result = classifier.classify_with_majority_voting(windows)
                        
                        if prediction_result:
                            last_prediction_result = prediction_result
                            is_processing = True
                            processing_start_time = time.time()
                            
                            # Log the result
                            if prediction_result['is_unknown']:
                                logging.info(f"Prediction: UNKNOWN (raw class: {prediction_result['predicted_class']})")
                            else:
                                gesture_name = GESTURE_NAMES.get(prediction_result['predicted_class'], 
                                                                f"Class {prediction_result['predicted_class']}")
                                logging.info(f"Prediction: {gesture_name} "
                                           f"(confidence: {prediction_result['max_confidence']:.2%}, "
                                           f"entropy: {prediction_result['entropy']:.2f}, "
                                           f"windows: {prediction_result['num_valid']}/{prediction_result['num_windows']})")
                    else:
                        # No valid windows after cleaning, just reset
                        logging.debug("Insufficient valid frames after cleaning, resetting")
                        sequence_buffer.reset("Insufficient valid frames")
            
            status = sequence_buffer.get_status()
            
            # Also process if we reach MAX_FRAMES (to prevent indefinite capture)
            if status['frame_count'] >= MAX_FRAMES and not is_processing:
                logging.debug(f"Reached MAX_FRAMES ({MAX_FRAMES}), triggering processing")
                windows = sequence_buffer.get_sliding_windows(MAX_SLIDING_WINDOWS)
                
                if windows:
                    logging.info(f"Processing {len(windows)} sliding windows (max frames reached)")
                    prediction_result = classifier.classify_with_majority_voting(windows)
                    
                    if prediction_result:
                        last_prediction_result = prediction_result
                        is_processing = True
                        processing_start_time = time.time()
                        
                        # Log the result
                        if prediction_result['is_unknown']:
                            logging.info(f"Prediction: UNKNOWN (raw class: {prediction_result['predicted_class']})")
                        else:
                            gesture_name = GESTURE_NAMES.get(prediction_result['predicted_class'], 
                                                            f"Class {prediction_result['predicted_class']}")
                            logging.info(f"Prediction: {gesture_name} "
                                       f"(confidence: {prediction_result['max_confidence']:.2%}, "
                                       f"entropy: {prediction_result['entropy']:.2f}, "
                                       f"windows: {prediction_result['num_valid']}/{prediction_result['num_windows']})")
        
        status = sequence_buffer.get_status()
        drawkit_seq.draw_overlay_continuous(
            frame, status, is_processing, processing_elapsed,
            last_prediction_result, GESTURE_NAMES, PROCESSING_DISPLAY_DURATION
        )
        
        # Draw FPS on frame
        drawkit_seq.draw_fps(current_fps, frame)

        cv2.imshow("i.MX Continuous Sequential Gesture Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="i.MX Continuous Sequential Gesture Recognition showcases the Machine "
        "Learning (ML) capabilities of the i.MX SoCs (i.MX 93 and i.MX 8M Plus) "
        "using the available Neural Processing Unit (NPU) to accelerate two "
        "Deep Learning vision-based models. This continuous mode uses sliding windows "
        "with majority voting for robust gesture recognition. The system captures frames "
        "continuously until hand is lost or maximum buffer is reached, then processes "
        "multiple overlapping sequences to make a final prediction."
    )

    parser.add_argument(
        "-f",
        "--file",
        metavar="file",
        type=str,
        help="Input file. It can be an image or a video.",
    )
    parser.add_argument(
        "-d",
        "--device",
        metavar="device",
        type=str,
        help="Camera device. Please provide the camera device " "as /dev/videoN.",
    )
    parser.add_argument(
        "-e",
        "--external_delegate_path",
        metavar="external delegate",
        type=str,
        help="Path to external delegate for HW acceleration.",
    )
    parser.add_argument(
        "--logging_level",
        metavar="logging level",
        type=int,
        default=logging.INFO,
        help="Logging level priority.",
    )

    parser.add_argument(
        "--palm_model",
        metavar="palm model",
        type=str,
        required=True,
        help="Path to palm detection model.",
    )
    parser.add_argument(
        "--hand_landmark_model",
        metavar="hand landmark model",
        type=str,
        required=True,
        help="Path to hand landmark model.",
    )
    parser.add_argument(
        "--classification_model",
        metavar="classification model",
        type=str,
        required=True,
        help="Path to sequential classification model.",
    )
    parser.add_argument(
        "--anchors",
        metavar="anchors",
        type=str,
        required=True,
        help="Path to anchors file.",
    )

    parser.add_argument(
        "--num_hands",
        metavar="Number of hands",
        type=int,
        default=1,
        help="Max number of hands that will be detected [1, 2]",
    )

    args = parser.parse_args()
    source = args.file
    if source:
        if not os.path.isfile(args.file):
            raise FileNotFoundError(
                "Source file does not exists. Please provide"
                " a valid source. You can check"
                " python3 main_seq.py --help for more details."
            )

    elif args.device:
        source = args.device

    if args.external_delegate_path:
        if not os.path.isfile(args.external_delegate_path):
            raise FileNotFoundError(f"File {args.external_delegate_path} not found.")

    if not args.num_hands in [1, 2]:
        raise ValueError("The Number of hands must be 1 or 2.")

    if not os.path.isfile(args.palm_model):
        raise FileNotFoundError(f"File {args.palm_model} not found.")
    if not os.path.isfile(args.hand_landmark_model):
        raise FileNotFoundError(f"File {args.hand_landmark_model} not found.")
    if not os.path.isfile(args.classification_model):
        raise FileNotFoundError(f"File {args.classification_model} not found.")
    if not os.path.isfile(args.anchors):
        raise FileNotFoundError(f"File {args.anchors} not found.")

    logging.basicConfig(level=args.logging_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger()

    logging.info("=== Continuous Sequential Gesture Recognition Configuration ===")
    logging.info("Source: %s", source)
    logging.info("Palm detection: %s", args.palm_model)
    logging.info("Hand landmark: %s", args.hand_landmark_model)
    logging.info("Classification model: %s", args.classification_model)
    logging.info("External delegate: %s", args.external_delegate_path)
    logging.info("Number of hands: %d", args.num_hands)
    logging.info("Min frames: %d, Max frames: %d", MIN_FRAMES, MAX_FRAMES)
    logging.info("Sequence length: %d, Max windows: %d", SEQUENCE_LENGTH, MAX_SLIDING_WINDOWS)
    logging.info("Processing display duration: %.1fs", PROCESSING_DISPLAY_DURATION)
    logging.info("===============================================================")

    run(source, args)