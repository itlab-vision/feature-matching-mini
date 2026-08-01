import cv2 as cv
import numpy as np
from abc import ABC, abstractmethod


class HPatchesTask(ABC):
    _TASKS = {}
    _img_indices = [2, 3, 4, 5, 6]

    def __init__(self, logger):
        self._logger = logger

    def __init_subclass__(cls, register=True, **kwargs):
        super().__init_subclass__(**kwargs)

        if register:
            key = cls.__name__.replace("Task", "").lower()
            if key:
                HPatchesTask._TASKS[key] = cls

    @classmethod
    def create(cls, task_name, logger, config=None):
        if config is None:
            config = {}

        task_class = cls._TASKS.get(task_name.lower())
        if not task_class:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(cls._TASKS.keys())}")

        return task_class(logger, config)

    @abstractmethod
    def eval_task(self, descriptors, split):
        pass

    @classmethod
    def _compute_tp_fp(cls, probs, targets, total_gt=None):
        probs = np.asarray(probs)
        targets = np.asarray(targets)

        n_pos = int(np.sum(targets == 1))
        n_neg = len(targets) - n_pos

        if total_gt is not None:
            if total_gt < n_pos:
                raise ValueError("total_gt must be >= n_pos")

            missing = total_gt - n_pos
            probs = np.concatenate([probs, np.full(missing, -np.inf)])
            targets = np.concatenate([targets, np.ones(missing)])
            n_pos = total_gt

        order = np.argsort(probs, kind='mergesort')[::-1]
        probs, targets = probs[order], targets[order]

        is_real = probs > -np.inf
        real_targets = targets[is_real]

        tp_curve = np.r_[0, np.cumsum(real_targets == 1)]
        fp_curve = np.r_[0, np.cumsum(real_targets == 0)]

        return tp_curve, fp_curve, n_pos, n_neg

    @classmethod
    def _precision_recall(cls, probs, targets, total_gt=None):
        tp, fp, p, n = cls._compute_tp_fp(probs, targets, total_gt)
        safe_div = 1e-12

        recall = tp / max(p, safe_div)
        precision = tp / np.maximum(tp + fp, safe_div)

        try:
            ap = np.trapezoid(precision, recall)
        except AttributeError:
            ap = np.trapz(precision, recall)

        return precision, recall, ap


class BaseMatchingTask(HPatchesTask, register=False):
    def __init__(self, logger, config):
        super().__init__(logger)

        self._eval_thresholds = config.get('eval_thresholds', [5.0])
        if not isinstance(self._eval_thresholds, list):
            self._eval_thresholds = [self._eval_thresholds]

    def _compute_numpos(self, data):
        kp_ref = data['kp_ref']
        kp_tgt = data['kp_tgt']
        H = data['H']
        tgt_shape = data['tgt_shape']

        if len(kp_ref) == 0 or len(kp_tgt) == 0:
            return np.array([], dtype=np.float32)

        pts_ref = np.array([kp.pt for kp in kp_ref], dtype=np.float32).reshape(-1, 1, 2)
        pts_tgt_gt = cv.perspectiveTransform(pts_ref, H).reshape(-1, 2)

        if tgt_shape is not None:
            h, w = tgt_shape[:2]
            valid_mask = ((pts_tgt_gt[:, 0] >= 0) & (pts_tgt_gt[:, 0] < w)
                          & (pts_tgt_gt[:, 1] >= 0) & (pts_tgt_gt[:, 1] < h))
            pts_tgt_gt = pts_tgt_gt[valid_mask]

        if len(pts_tgt_gt) == 0:
            return np.array([], dtype=np.float32)

        pts_tgt = np.array([kp.pt for kp in kp_tgt], dtype=np.float32)

        diff = pts_tgt_gt[:, np.newaxis, :] - pts_tgt[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        return dist_matrix.min(axis=1)

    @staticmethod
    def _numpos(min_dists, threshold):
        if min_dists.size == 0:
            return 0
        return int(np.sum(min_dists <= threshold))

    def _compute_distances(self, data):
        kp_ref = data['kp_ref']
        kp_tgt = data['kp_tgt']
        matches = data['matches']
        H = data['H']

        if not matches:
            return None

        pts_ref = np.array([kp_ref[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        pts_tgt_pred = np.array([kp_tgt[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        pts_tgt_gt = cv.perspectiveTransform(pts_ref, H)

        pixel_dists = np.linalg.norm(pts_tgt_pred - pts_tgt_gt, axis=2).flatten()
        descriptor_dists = np.array([m.distance for m in matches], dtype=np.float32)

        if descriptor_dists.size > 0:
            max_dist = descriptor_dists.max()
            scores = 1.0 - (descriptor_dists / max_dist)
        else:
            scores = np.array([], dtype=np.float32)

        return {
            'pixel_dists': pixel_dists,
            'num_kp_ref': len(kp_ref),
            'num_kp_tgt': len(kp_tgt),
            'num_matches': len(matches),
            'scores': scores
        }

    @staticmethod
    def _match_results(dist_result, threshold):
        labels = dist_result['pixel_dists'] <= threshold
        return {
            'labels': labels,
            'distances': dist_result['pixel_dists'],
            'num_kp_ref': dist_result['num_kp_ref'],
            'num_kp_tgt': dist_result['num_kp_tgt'],
            'num_matches': dist_result['num_matches'],
            'scores': dist_result['scores']
        }


class MatchingAPTask(BaseMatchingTask):
    def eval_task(self, matching_data, split):
        results = {threshold: {seq: {} for seq in split} for threshold in self._eval_thresholds}
        self._logger.info(f'Evaluating Feature Matching (mAP) @ {self._eval_thresholds}px')

        for seq in split:
            if seq not in matching_data:
                self._logger.warning(f"Scene '{seq}' not found in matching_data."
                                     f"Skipping for threshold {self._eval_thresholds}px")
                continue

            for i in self._img_indices:
                data = matching_data[seq].get(i)
                if not data:
                    self._logger.warning(f"Image pair 1-{i} in scene '{seq}' has no matching data. "
                                         f"Skipping for all thresholds.")
                    continue

                dist_result = self._compute_distances(data)
                if dist_result is None:
                    self._logger.warning(f"Could not compute distances for image pair 1-{i} in scene '{seq}'."
                                         f"Matches list is empty.")
                    continue

                min_dists = self._compute_numpos(data)
                for threshold in self._eval_thresholds:
                    res = self._match_results(dist_result, threshold)
                    numpos = self._numpos(min_dists, threshold)
                    _, _, ap = self._precision_recall(res['scores'], res['labels'], total_gt=numpos)
                    results[threshold][seq][i] = {'ap': ap}

        return results

    def report_metrics(self, results, task_name="Feature Matching (mAP)"):
        metrics = {}

        for threshold, threshold_results in results.items():
            all_ap_values = [
                img_data['ap']
                for scene_data in threshold_results.values()
                for img_data in scene_data.values()
                if 'ap' in img_data
            ]

            if not all_ap_values:
                self._logger.warning(f"No AP results found for threshold {threshold}px")
                metrics[threshold] = None
                continue

            mean_total_ap = float(np.mean(all_ap_values))
            self._logger.info(f"{task_name.upper()} @ {threshold}px, MAP: {mean_total_ap:.4f}")
            metrics[threshold] = {'mean_ap': mean_total_ap}

        return metrics


class MatchingScoreTask(BaseMatchingTask):
    def eval_task(self, matching_data, split):
        results = {threshold: {seq: {} for seq in split} for threshold in self._eval_thresholds}
        self._logger.info(f'Evaluating Matching Score & Precision @ {self._eval_thresholds}px')

        for seq in split:
            if seq not in matching_data:
                self._logger.warning(f"Scene '{seq}' not found in matching_data."
                                     f"Skipping for threshold {self._eval_thresholds}px")
                continue

            for i in self._img_indices:
                data = matching_data[seq].get(i)
                if not data:
                    self._logger.warning(f"Image pair 1-{i} in scene '{seq}' has no matching data. "
                                         f"Skipping for all thresholds.")
                    continue

                dist_result = self._compute_distances(data)
                if dist_result is None:
                    self._logger.warning(f"Could not compute distances for image pair 1-{i} in scene '{seq}'. "
                                         f"Matches list is empty.")
                    continue

                for threshold in self._eval_thresholds:
                    res = self._match_results(dist_result, threshold)
                    num_inliers = np.sum(res['labels'])
                    results[threshold][seq][i] = {
                        'ms': num_inliers / min(res['num_kp_ref'], res['num_kp_tgt']),
                        'prec': num_inliers / res['num_matches'] if res['num_matches'] > 0 else 0
                    }

        return results

    def report_metrics(self, results, task_name="Matching Score & Precision"):
        metrics = {}

        for threshold, threshold_results in results.items():
            all_ms_values = [
                img_data['ms']
                for scene_data in threshold_results.values()
                for img_data in scene_data.values()
                if 'ms' in img_data
            ]

            if not all_ms_values:
                self._logger.warning(f"No MS results for threshold {threshold}px")
                metrics[threshold] = None
                continue

            all_prec_values = [
                img_data['prec']
                for scene_data in threshold_results.values()
                for img_data in scene_data.values()
                if 'prec' in img_data
            ]

            mean_total_ms = float(np.mean(all_ms_values))
            mean_total_prec = float(np.mean(all_prec_values))

            self._logger.info(f"{task_name.upper()} @ {threshold}px, MS: {mean_total_ms:.4f},"
                              f" Prec: {mean_total_prec:.4f}")
            metrics[threshold] = {'mean_ms': mean_total_ms, 'mean_prec': mean_total_prec}

        return metrics


class HomographyAUCTask(HPatchesTask):
    _HOMOGRAPHY_METHODS = {
        "ransac": cv.RANSAC,
        "magsac": cv.USAC_MAGSAC,
        "lmeds": cv.LMEDS,
        "rho": cv.RHO
    }

    def __init__(self, logger, config):
        super().__init__(logger)

        self._eval_thresholds = config.pop('eval_thresholds', [5.0])
        if not isinstance(self._eval_thresholds, list):
            self._eval_thresholds = [self._eval_thresholds]

        self._homography_threshold = config.pop('homography_threshold', 3.0)
        self._homography_method = config.pop('homography_method', "ransac")

    def eval_task(self, matching_data, split):
        results = {threshold: {seq: {} for seq in split} for threshold in self._eval_thresholds}
        self._logger.info(f'Evaluating Homography AUC @ {self._eval_thresholds}px')

        for threshold in self._eval_thresholds:
            for seq in split:
                if seq not in matching_data:
                    self._logger.warning(f"Scene '{seq}' not found in matching_data."
                                         f" Skipping for threshold {threshold}px")
                    continue

                for i in self._img_indices:
                    data = matching_data[seq].get(i)
                    if not data or not data['matches']:
                        self._logger.warning(f"Image pair 1-{i} in scene '{seq}' has no matching data. "
                                             f"Skipping for all thresholds.")
                        continue

                    kp_ref = data['kp_ref']
                    kp_tgt = data['kp_tgt']
                    matches = data['matches']

                    pts_ref = np.array([kp_ref[m.queryIdx].pt for m in matches],
                                       dtype=np.float32).reshape(-1, 1, 2)
                    pts_tgt_pred = np.array([kp_tgt[m.trainIdx].pt for m in matches],
                                            dtype=np.float32).reshape(-1, 1, 2)

                    if len(pts_ref) < 4:
                        self._logger.warning(f"Not enough matches ({len(pts_ref)}) to compute homography "
                                             f"for scene '{seq}' image pair 1-{i}. Minimum 4 required.")
                        continue

                    H_gt = data['H']
                    H_pred, mask = cv.findHomography(pts_ref, pts_tgt_pred,
                                                     self._HOMOGRAPHY_METHODS[self._homography_method],
                                                     self._homography_threshold)

                    if H_pred is None:
                        self._logger.warning(f"Homography estimation failed for scene '{seq}' image pair 1-{i}. "
                                             f"Check points distribution.")
                        continue

                    h, w = data['ref_shape'][:2]
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

                    corners_gt = cv.perspectiveTransform(corners, H_gt)
                    corners_pred = cv.perspectiveTransform(corners, H_pred)

                    error = np.mean(np.linalg.norm(corners_gt - corners_pred, axis=2))
                    results[threshold][seq][i] = {'error': error}

        return results

    def report_metrics(self, results, task_name="Homography AUC"):
        metrics = {}

        for threshold, threshold_results in results.items():
            all_errors = [
                img['error']
                for s in threshold_results.values()
                for img in s.values()
                if 'error' in img
            ]

            if not all_errors:
                self._logger.warning(f"No results for threshold {threshold}px")
                metrics[threshold] = None
                continue

            thresholds_lin = np.linspace(0, threshold, 100)
            acc_curve = [np.mean(np.array(all_errors) < t) for t in thresholds_lin]
            global_auc = float(np.trapezoid(acc_curve, thresholds_lin) / threshold)

            self._logger.info(f"{task_name.upper()} @ {threshold}px, AUC: {global_auc:.4f}")
            metrics[threshold] = {'mean_auc': global_auc}

        return metrics
