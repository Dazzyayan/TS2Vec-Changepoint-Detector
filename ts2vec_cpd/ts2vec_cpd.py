import numpy as np
from scipy.signal import find_peaks

from ts2vec import TS2Vec
# Z. Yue et al., ‘TS2Vec: Towards Universal Representation of Time Series’, AAAI, vol. 36, no. 8, pp. 8980–8987, Jun. 2022, doi: 10.1609/aaai.v36i8.20881.


def train_ts2vec_model(train_data, input_dims=None, output_dims=320, device=0):
    """
    Trains a TS2Vec model on the provided training data.

    Args:
        train_data (np.ndarray): Training data (n_instances, n_timestamps, n_features).
        input_dims (int): Input dimensions for TS2Vec model.
        output_dims (int): Output dimensions for TS2Vec representations.
        device (int): Device to use for training (0 for GPU, -1 for CPU).

    Returns:
        TS2Vec: The trained TS2Vec model.
    """
    print("Training TS2Vec model...")
    if input_dims is None:
        input_dims = train_data.shape[-1]
        
    model = TS2Vec(
        input_dims=input_dims,
        device=device,
        output_dims=output_dims
    )
    _ = model.fit(
        train_data,
        verbose=False
    )
    print("TS2Vec model training complete.")
    return model


class TS2VecChangepointDetector:
    """TS2Vec-based changepoint detector."""

    def __init__(self, model, sliding_length=1, sliding_padding=6,
                 local_window_size=6, prominence_threshold=1.0):
        """
        Initialise TS2Vec changepoint detector.

        Args:
            model: Trained TS2Vec model
            sliding_length: The two sliding parameters influence how data is batched for training.  
            sliding_padding: 
            local_window_size: Size of local window for maxpooling and window search
            prominence_threshold: Minimum prominence for peak detection
        """
        self.model = model
        self.sliding_length = sliding_length
        self.sliding_padding = sliding_padding
        self.local_window_size = local_window_size
        self.l2_prominence_threshold = prominence_threshold
        self.detected_cps = None

    def _calculate_l2_distance(self, vec1, vec2):
        """Calculate L2 distance between vectors."""
        return np.linalg.norm(vec1 - vec2)

    def detect_changepoints(self, data):
        """
        Detect changepoints in the data.
        data must be in the shape of (1, n_timestamps, n_features) and all missing data should be set to NaN.
        """
        # ensuring the data is in the correct shape for encoder
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim == 1:
            data = data[np.newaxis, :, np.newaxis]

        # encoding the data using the TS2Vec model
        unmasked_representations = self.model.encode(
            data, causal=True, sliding_length=self.local_window_size,
            sliding_padding=self.sliding_padding, mask=None,
            encoding_window=self.local_window_size
        )
        unmasked_instance_repr = unmasked_representations.squeeze(0)

        instance_l2_distances = []
        N = self.local_window_size
        n_repr = unmasked_instance_repr.shape[0]

        # As we set encoding_window to local_window_size, we can compute the L2 distance between each timestamp
        # and its N neighbors without having to aggregate the data ourselves as TS2Vec does this for us.
        for t in range(n_repr):
            # getting maxpooled representations
            left_idx = t
            right_idx = min(n_repr - 1, t + N)
            left_repr = unmasked_instance_repr[left_idx, :]
            right_repr = unmasked_instance_repr[right_idx, :]
            
            # compute L2 distance between the two representations
            l2_distance = self._calculate_l2_distance(left_repr, right_repr)
            instance_l2_distances.append(l2_distance)

        l2_distances = np.array(instance_l2_distances)
        filtered_l2_distances = triangular_filter(l2_distances, window_size=self.local_window_size)
        valid_indices = ~np.isnan(filtered_l2_distances)
        valid_signal = filtered_l2_distances[valid_indices]

        detected_changepoints = []

        if len(valid_signal) > 0:
          peaks, _ = find_peaks(valid_signal, prominence=self.l2_prominence_threshold)
          if len(peaks) > 0:
              original_indices = np.where(valid_indices)[0]
              detected_changepoints = [original_indices[peak_idx] for peak_idx in peaks]

        self.detected_cps = detected_changepoints
        return detected_changepoints, filtered_l2_distances

    def evaluate_performance(self, ground_truth_cps, detected_cps=None, tolerance=20):
        """Evaluate performance."""
        if detected_cps is None:
            detected_cps = self.detected_cps or []
        return evaluate_changepoint_detection(ground_truth_cps, detected_cps, tolerance)
    
    
    
def triangular_filter(signal, window_size):
    """
    Triangular filter for dissimilarity measure smoothing
    Adapted from:
    T. De Ryck, M. De Vos and A. Bertrand, "Change Point Detection in Time Series Data Using Autoencoders With a Time-Invariant Representation," in IEEE Transactions on Signal Processing, vol. 69, pp. 3513-3524, 2021
    """
    mask = np.ones((2*window_size+1,))
    for i in range(window_size):
        mask[i] = i/(window_size**2)
        mask[-(i+1)] = i/(window_size**2)
    mask[window_size] = window_size/(window_size**2)

    signal_out = np.zeros(np.shape(signal))

    if len(np.shape(signal)) >1:
        for i in range(np.shape(signal)[1]):
            signal_extended = np.concatenate((signal[0,i]*np.ones(window_size), signal[:,i], signal[-1,i]*np.ones(window_size)))
            signal_out[:,i] = np.convolve(signal_extended, mask, 'valid')
    else:
        signal = np.concatenate((signal[0]*np.ones(window_size), signal, signal[-1]*np.ones(window_size)))
        signal_out = np.convolve(signal, mask, 'valid')

    return signal_out


def evaluate_changepoint_detection(ground_truth_cps, detected_cps, tolerance=20, closest_match=True):
    """
    Evaluates changepoint detection performance for a single instance.

    Args:
        ground_truth_cps (list): List of ground truth changepoint timestamps.
        detected_cps (list): List of detected changepoint timestamps.
        tolerance (int): The tolerance window around ground truth changepoints
                         to consider a detected changepoint a True Positive.
        closest_match (bool): If True, each detected changepoint matches only the closest
                             ground truth within tolerance.

    Returns:
        tuple: (tp, fp, fn, precision, recall, f1_score)
               tp (int): True Positives
               fp (int): False Positives
               fn (int): False Negatives
               precision (float): Precision score
               recall (float): Recall score
               f1_score (float): F1-score
    """
    tp = 0
    fp = 0
    fn = 0

    # Handle edge case: no ground truth changepoints
    if len(ground_truth_cps) == 0:
        # All detected changepoints are false positives
        fp = len(detected_cps)
        fn = 0  # No ground truth means no false negatives

        # Calculate metrics for no ground truth case
        precision = 0.0 if fp > 0 else 1.0  # Perfect precision only if no detections
        recall = 1.0  # Perfect recall when no ground truth to miss
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return tp, fp, fn, precision, recall, f1_score

    # Create a boolean array to track if a ground truth CP has been matched
    matched_gt = [False] * len(ground_truth_cps)

    # closest match prevents a single detected CP from matching to multiple ground truth CPs
    if closest_match:
        matched_detected = [False] * len(detected_cps)
        
        for i, detected_cp in enumerate(detected_cps):
            if matched_detected[i]:
                continue  # Skip if already matched
                
            best_match_idx = -1
            min_distance = float('inf')
            
            # Find the closest unmatched ground truth CP within tolerance
            for j, gt_cp in enumerate(ground_truth_cps):
                if matched_gt[j]:
                    continue  # Skip if already matched
                    
                distance = abs(detected_cp - gt_cp)
                if distance <= tolerance and distance < min_distance:
                    min_distance = distance
                    best_match_idx = j
            
            # If a match is found, mark both as matched and count as TP
            if best_match_idx != -1:
                tp += 1
                matched_gt[best_match_idx] = True
                matched_detected[i] = True
            else:
                fp += 1
    else:
        # Simple matching
        for detected_cp in detected_cps:
            is_tp = False
            for i, gt_cp in enumerate(ground_truth_cps):
                if abs(detected_cp - gt_cp) <= tolerance and not matched_gt[i]:
                    tp += 1
                    matched_gt[i] = True
                    is_tp = True
                    break  # Move to the next detected CP once a match is found
            if not is_tp:
                fp += 1

    # Count False Negatives (ground truth CPs that were not matched)
    fn = len(ground_truth_cps) - sum(matched_gt)

    # Calculate Precision, Recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return tp, fp, fn, precision, recall, f1_score