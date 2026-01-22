import cv2
import numpy as np
import tensorflow.lite as tflite
import csv


class Bbox:
    """Represents a bounding box with center, dimensions and rotation"""
    def __init__(self, center, dims, rotation=0.0):
        self.center = center
        self.dims = dims
        self.rotation = rotation


class PalmBbox(Bbox):
    """Represents a bounding box that encloses a palm instance"""
    def __init__(self, center, dims, score, keypoints, rotation=0.0):
        super().__init__(center, dims, rotation)
        self.score = score
        self.keypoints = keypoints
    
    def coordinates(self):
        """Return bounding box coordinates in [xmin, ymin, xmax, ymax] format"""
        x_min = self.center[0] - self.dims[0] / 2.0
        y_min = self.center[1] - self.dims[1] / 2.0
        x_max = self.center[0] + self.dims[0] / 2.0
        y_max = self.center[1] + self.dims[1] / 2.0
        return [x_min, y_min, x_max, y_max]
    
    def compute_rotation(self):
        """Return the rotation value of a palm bbox instance"""
        x_0, y_0 = self.keypoints[0]  # Wrist
        x_1, y_1 = self.keypoints[2]  # Middle finger MCP
        
        target_angle = np.pi * 0.5
        rot = target_angle - np.arctan2(-(y_1 - y_0), (x_1 - x_0))
        norm_rot = rot - 2 * np.pi * np.floor((rot + np.pi) / (2 * np.pi))
        return norm_rot


class LandmarkBbox(Bbox):
    """Represents a bounding box that encloses a hand instance using landmarks"""
    def __init__(self, center, dims, score, handedness, landmarks, rotation=0.0):
        super().__init__(center, dims, rotation)
        self.score = score
        self.handedness = handedness
        self.landmarks = landmarks
    
    def compute_rotation(self):
        """Return the rotation value of a LandmarkBbox instance"""
        x_0, y_0 = self.landmarks[0][:2]  # Wrist (x, y)
        x_1, y_1 = self.landmarks[9][:2]  # Middle finger MCP (x, y)
        
        target_angle = np.pi * 0.5
        rot = target_angle - np.arctan2(-(y_1 - y_0), (x_1 - x_0))
        norm_rot = rot - 2 * np.pi * np.floor((rot + np.pi) / (2 * np.pi))
        return norm_rot


class PalmDetector:
    """Performs palm detection in an input image"""
    
    def __init__(self, model, anchors, palm_detection_conf=0.5, min_suppr_thr=0.3, num_palms=2):
        self._interpreter = tflite.Interpreter(model_path=model)
        self._interpreter.allocate_tensors()
        
        self._palm_detection_conf = palm_detection_conf
        self._min_supression_threshold = min_suppr_thr
        self._num_palms = num_palms
        
        # Read SSD anchors
        with open(anchors, "r", encoding="utf-8") as csv_file:
            self._anchors = np.array([line for line in csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)])
        
        out_details = self._interpreter.get_output_details()
        in_details = self._interpreter.get_input_details()
        
        self._in_idx = in_details[0]["index"]
        self._out_reg_idx = out_details[1]["index"]
        self._out_clf_idx = out_details[0]["index"]
        
        # Warm-up
        batch, width, height, channel = tuple(in_details[0]["shape"].tolist())
        self._interpreter.set_tensor(self._in_idx, np.random.rand(batch, width, height, channel).astype("float32"))
        self._interpreter.invoke()
    
    def sigmoid(self, array):
        """Applies sigmoid function to an array"""
        return 1.0 / (1.0 + np.exp(-array))
    
    def _inter_over_union(self, bbox1, bbox2):
        """Calculates intersection over union score between two boxes"""
        if bbox1 is None or bbox2 is None:
            return 0.0
        
        x1_min, y1_min, x1_max, y1_max = bbox1.coordinates()
        x2_min, y2_min, x2_max, y2_max = bbox2.coordinates()
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        x3_min = max(x1_min, x2_min)
        x3_max = min(x1_max, x2_max)
        y3_min = max(y1_min, y2_min)
        y3_max = min(y1_max, y2_max)
        
        intersect_area = max(0, x3_max - x3_min) * max(0, y3_max - y3_min)
        denominator = box1_area + box2_area - intersect_area
        return intersect_area / denominator if denominator > 0.0 else 0.0
    
    def _non_max_suppression(self, bboxes, min_suppression_threshold):
        """Non-maximum suppression algorithm"""
        bboxes = sorted(bboxes, key=lambda bbox: bbox.score, reverse=True)
        kept_bboxes = []
        
        for bbox in bboxes:
            suppressed = False
            for kept_bbox in kept_bboxes:
                similarity = self._inter_over_union(kept_bbox, bbox)
                if similarity > min_suppression_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept_bboxes.append(bbox)
        return kept_bboxes
    
    def __call__(self, frame):
        """Performs palm detection in an input image"""
        self._interpreter.set_tensor(self._in_idx, frame[None])
        self._interpreter.invoke()
        
        out_reg = self._interpreter.get_tensor(self._out_reg_idx)[0]
        out_clf = self._interpreter.get_tensor(self._out_clf_idx)[0, :, 0]
        
        scores = self.sigmoid(out_clf)
        mask = scores > self._palm_detection_conf
        
        detections = out_reg[mask]
        anchors = self._anchors[mask]
        scores = scores[mask]
        
        if detections.shape[0] == 0:
            return None
        
        bboxes = []
        for detection, anchor, score in zip(detections, anchors, scores):
            deltax, deltay, width, height = detection[0:4] / frame.shape[0]
            anchor_center = anchor[0:2]
            keypoints = anchor_center + detection[4:].reshape(-1, 2)
            
            center = [anchor_center[0] + deltax, anchor_center[1] + deltay]
            dims = [width, height]
            
            bbox = PalmBbox(center, dims, score, keypoints)
            bboxes.append(bbox)
        
        bboxes = self._non_max_suppression(bboxes, self._min_supression_threshold)
        return bboxes[:self._num_palms]


class HandLandmark:
    """Performs the detection of the hand landmarks in an input image"""
    
    def __init__(self, model):
        self._interpreter = tflite.Interpreter(model_path=model)
        self._interpreter.allocate_tensors()
        
        out_details = self._interpreter.get_output_details()
        in_details = self._interpreter.get_input_details()
        
        self._in_idx = in_details[0]["index"]
        self._out_lmks_idx = out_details[3]["index"]
        self._out_score_idx = out_details[2]["index"]
        self._out_handness_idx = out_details[0]["index"]
        
        # Warm-up
        batch, width, height, channel = tuple(in_details[0]["shape"].tolist())
        self._interpreter.set_tensor(self._in_idx, np.random.rand(batch, width, height, channel).astype("float32"))
        self._interpreter.invoke()
    
    def __call__(self, frame):
        """Performs the detection of the hand landmarks in an input image"""
        self._interpreter.set_tensor(self._in_idx, frame[None])
        self._interpreter.invoke()
        
        landmarks = self._interpreter.get_tensor(self._out_lmks_idx)[0] / 224
        handedness = self._interpreter.get_tensor(self._out_handness_idx)[0]
        score = self._interpreter.get_tensor(self._out_score_idx)[0]
        
        # Keep all 3 components (x, y, z)
        landmarks = landmarks.reshape(-1, 3)
        
        lm_bbox = LandmarkBbox(
            center=None,
            dims=None,
            rotation=None,
            score=float(score),
            handedness=float(handedness),
            landmarks=landmarks,
        )
        
        return lm_bbox


class HandTracker:
    
    def __init__(self, palm_detection_model, hand_landmark_model, anchors, 
                 num_hands=2, palm_detection_conf=0.5, hand_landmark_conf=0.49):
        
        self._palm_detector = PalmDetector(
            model=palm_detection_model,
            anchors=anchors,
            palm_detection_conf=palm_detection_conf,
            min_suppr_thr=0.3,
            num_palms=num_hands
        )
        
        self._hand_landmarks = HandLandmark(model=hand_landmark_model)
        
        self._hand_landmark_conf = hand_landmark_conf
        self._num_hands = num_hands
        self._previous_hand_bboxes = []
    
    def _normalize_image(self, image):
        """Returns an image in the range [-1.0, 1.0]"""
        return np.ascontiguousarray((image / 255.0).astype("float32"))
    
    def _preprocess(self, image, dim):
        """Preprocesses an image for model inference"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shape = image.shape
        
        pad_size = ((max(shape) - shape[0]) // 2, (max(shape) - shape[1]) // 2)
        padded_image = np.pad(
            image,
            ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (0, 0)),
            mode="constant",
        )
        
        resized_image = cv2.resize(padded_image, (dim, dim))
        norm_image = self._normalize_image(np.ascontiguousarray(resized_image))
        padding = {"pad_size": pad_size, "pad_img_dim": padded_image.shape[0]}
        
        return norm_image, padding
    
    def _compute_hand_bbox(self, bbox, shift, box_enlarge):
        """Calculate a bbox that encloses an entire hand"""
        if isinstance(bbox, LandmarkBbox):
            palm_joints = [0, 1, 2, 5, 9, 13, 17]
            max_x = max_y = -np.inf
            min_x = min_y = np.inf
            
            for joint in palm_joints:
                max_x = max(bbox.landmarks[joint][0], max_x)
                max_y = max(bbox.landmarks[joint][1], max_y)
                min_x = min(bbox.landmarks[joint][0], min_x)
                min_y = min(bbox.landmarks[joint][1], min_y)
            
            center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
            dims = (max_x - min_x, max_y - min_y)
            bbox.center = center
            bbox.dims = dims
        
        shift_x, shift_y = shift
        rotation = bbox.compute_rotation()
        width, height = bbox.dims
        
        d_x = (width * shift_x) * np.cos(rotation) - (height * shift_y) * np.sin(rotation)
        d_y = (width * shift_x) * np.sin(rotation) + (height * shift_y) * np.cos(rotation)
        
        long_side = np.max([width, height])
        hand_w = long_side * box_enlarge
        hand_h = long_side * box_enlarge
        hand_center = [bbox.center[0] + d_x, bbox.center[1] + d_y]
        
        hand_bbox = Bbox(hand_center, [hand_w, hand_h], rotation)
        return hand_bbox
    
    def _scale_bbox_to_frame(self, bbox, padding):
        """Scale bbox coordinates to the size of a frame"""
        img_dim = padding["pad_img_dim"]
        pad_h, pad_w = padding["pad_size"]
        
        if isinstance(bbox, LandmarkBbox):
            # Scale x and y, keep z as is (already normalized)
            bbox.landmarks[:, 0] = bbox.landmarks[:, 0] * img_dim
            bbox.landmarks[:, 1] = bbox.landmarks[:, 1] * img_dim
            return bbox
        
        bbox.center[0] = bbox.center[0] * img_dim - pad_w
        bbox.center[1] = bbox.center[1] * img_dim - pad_h
        bbox.dims[0] = bbox.dims[0] * img_dim
        bbox.dims[1] = bbox.dims[1] * img_dim
        
        return bbox
    
    def _crop_rotated_rectangle(self, image, rect):
        """Crop rotated rectangle from image"""
        box = cv2.boxPoints(rect).astype("float32")
        width, height = int(rect[1][0]), int(rect[1][1])
        
        dst_pts = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype="float32",
        )
        
        mat = cv2.getPerspectiveTransform(box, dst_pts)
        m_inv = cv2.getPerspectiveTransform(dst_pts, box)
        
        cropped = cv2.warpPerspective(image, mat, (width, height))
        return cropped, m_inv
    
    def __call__(self, frame):
        """Detects and tracks hands in a given frame"""
        detections = []
        
        # Run palm detector if needed
        if len(self._previous_hand_bboxes) < self._num_hands:
            in_frame, padding = self._preprocess(frame, 192)
            palm_bboxes = self._palm_detector(in_frame)
            
            if palm_bboxes:
                for i in range(self._num_hands - len(self._previous_hand_bboxes)):
                    if i >= len(palm_bboxes):
                        break
                    palm_bbox = palm_bboxes[i]
                    
                    hand_bbox = self._compute_hand_bbox(palm_bbox, (0.0, -0.5), 2.6)
                    hand_bbox = self._scale_bbox_to_frame(hand_bbox, padding)
                    self._previous_hand_bboxes.append(hand_bbox)
        
        # Process existing hand bboxes
        if self._previous_hand_bboxes:
            new_hand_bboxes = []
            for hand_bbox in self._previous_hand_bboxes:
                center = hand_bbox.center
                dims = hand_bbox.dims
                rot_degrees = hand_bbox.rotation * 180 / np.pi
                
                rect = (center, dims, rot_degrees)
                cropped, m_inv = self._crop_rotated_rectangle(frame, rect)
                
                norm_cropped, padding = self._preprocess(cropped, 224)
                lm_bbox = self._hand_landmarks(norm_cropped)
                
                if lm_bbox.score < self._hand_landmark_conf:
                    continue
                
                lm_bbox = self._scale_bbox_to_frame(lm_bbox, padding)
                
                # Transform x and y coordinates, keep z as is
                landmarks_xy = lm_bbox.landmarks[:, :2].reshape(-1, 1, 2)
                transformed_xy = cv2.perspectiveTransform(landmarks_xy, m_inv)
                lm_bbox.landmarks[:, :2] = transformed_xy.reshape(-1, 2)
                
                hand_bbox = self._compute_hand_bbox(lm_bbox, (0.0, -0.3), 2.6)
                
                detections.append((lm_bbox.landmarks, hand_bbox, lm_bbox.handedness))
                new_hand_bboxes.append(hand_bbox)
            
            self._previous_hand_bboxes = new_hand_bboxes
        
        return detections