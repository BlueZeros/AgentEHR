from typing import Optional, Dict, Any, List
import numpy as np



class TrajectoryJudge:
    """
    Score-based Judge for classifying trajectory success/failure.

    Uses historical F1, precision, and recall scores to determine success.
    Supports multiple runs aggregation and historical comparison strategies.
    """

    def __init__(
        self, 
        score_threshold: float = 0.2,
        use_historical_comparison: bool = False,
        f1_weight: float = 0.4,
        precision_weight: float = 0.2,
        recall_weight: float = 0.4,
        use_weighted_score: bool = True
    ):
        """
        Args:
            score_threshold: Minimum F1 score to consider success (default: 0.2)
            use_historical_comparison: Compare with historical successful cases (default: False)
            f1_weight: Weight for F1 in weighted scoring (default: 0.4)
            precision_weight: Weight for precision in weighted scoring (default: 0.2)
            recall_weight: Weight for recall in weighted scoring (default: 0.4)
            use_weighted_score: Use weighted combination of F1/precision/recall (default: True)
        """
        self.score_threshold = score_threshold
        self.f1_weight = f1_weight
        self.precision_weight = precision_weight
        self.recall_weight = recall_weight
        self.use_weighted_score = use_weighted_score
        self.use_historical_comparison = use_historical_comparison

    def judge_trajectory_success(self, score, scores_list) -> bool:
        final_score = self.calculate_score(score)

        # Strategy: Compare with historical data if enabled
        if self.use_historical_comparison:
            # Must meet both the threshold AND be better than history
            success = (final_score >= self.score_threshold) and self._compare_with_history(final_score, scores_list)
        else:
            # Default: Simple threshold comparison
            success = final_score >= self.score_threshold
                
        return success

    def calculate_score(self, score) -> float:
        f1_score = score.get('f1_score', 0.0)
        precision = score.get('precision', 0.0)
        recall = score.get('recall', 0.0)
        if self.use_weighted_score:
            return (
                self.f1_weight * f1_score +
                self.precision_weight * precision +
                self.recall_weight * recall
            )
        else:
            return f1_score
    
    def _compare_with_history(self, score, scores_list) -> bool:
        final_scores = []
        for scores in scores_list:
            final_scores.append(self.calculate_score(scores))
        average_score = np.mean(final_scores)
        return score >= average_score
