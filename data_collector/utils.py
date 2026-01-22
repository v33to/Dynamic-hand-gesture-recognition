import numpy as np

def compute_dtw_distance(seq1, seq2):
    """
    Computes the Dynamic Time Warping (DTW) distance between two sequences.
    """
    n, m = len(seq1), len(seq2)
    
    # Flatten checks to ensure inputs are (frames, features)
    if seq1.ndim == 1:
        seq1 = seq1.reshape(n, -1)
    if seq2.ndim == 1:
        seq2 = seq2.reshape(m, -1)
        
    # Efficient broadcasting to compute Euclidean distance matrix
    diff = seq1[:, np.newaxis, :] - seq2[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    
    # Initialize DTW matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_matrix[i-1, j-1]
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # Insertion
                                          dtw_matrix[i, j-1],    # Deletion
                                          dtw_matrix[i-1, j-1])  # Match

    # Normalized distance
    return dtw_matrix[n, m] / (n + m)

def calculate_sequence_similarity(seq1, seq2, sigma=0.5):
    """
    Calculate similarity score (0 to 1) using DTW.
    """
    distance = compute_dtw_distance(seq1, seq2)
    # Exponential kernel gives a smoother 0-1 score than simple division
    similarity = np.exp(-distance / sigma)
    return similarity

def calculate_dataset_diversity(new_sequence, dataset_sequences, pending_sequences):
    """
    Calculate diversity considering both existing dataset and pending sequences.
    """
    all_sequences = dataset_sequences + pending_sequences
    
    if not all_sequences:
        return 1.0, []
    
    similarities = []
    # Reshape new sequence once to ensure 2D structure
    new_seq_flat = np.array(new_sequence)
    if new_seq_flat.ndim == 1:
         new_seq_flat = new_seq_flat.reshape(-1, 63)

    for seq in all_sequences:
        comp_seq = np.array(seq)
        if comp_seq.ndim == 1:
            comp_seq = comp_seq.reshape(-1, 63)
            
        sim = calculate_sequence_similarity(new_seq_flat, comp_seq)
        similarities.append(sim)
    
    if not similarities:
        return 1.0, []
    
    max_similarity = max(similarities)
    diversity_score = 1.0 - max_similarity
    
    return diversity_score, similarities


def rotate_sequence(sequence, angle_degrees, num_landmarks=21):
    """
    Rotate the entire pose in each frame by the specified angle.
    
    Args:
        sequence: numpy array of shape (num_frames, num_landmarks * 3) or (num_frames, num_landmarks, 3)
        angle_degrees: rotation angle in degrees (positive = counter-clockwise)
        num_landmarks: number of landmarks per frame (default: 21)
    
    Returns:
        Rotated sequence with same shape as input
    """
    original_shape = sequence.shape
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotation matrix for 2D rotation (around Z-axis in XY plane)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    # Ensure sequence is 2D (frames, features)
    if sequence.ndim == 3:
        sequence = sequence.reshape(sequence.shape[0], -1)
    
    rotated_sequence = sequence.copy()
    num_frames = sequence.shape[0]
    
    for frame_idx in range(num_frames):
        # Reshape frame to (num_landmarks, 3)
        frame = sequence[frame_idx].reshape(num_landmarks, 3)
        
        # Calculate centroid of the hand in this frame
        centroid = frame.mean(axis=0)
        
        # Center the landmarks
        centered = frame - centroid
        
        # Apply rotation
        rotated = centered @ rotation_matrix.T
        
        # Move back to original position
        rotated = rotated + centroid
        
        # Store back in the sequence
        rotated_sequence[frame_idx] = rotated.flatten()
    
    # Restore original shape if needed
    if len(original_shape) == 3:
        rotated_sequence = rotated_sequence.reshape(original_shape)
    
    return rotated_sequence