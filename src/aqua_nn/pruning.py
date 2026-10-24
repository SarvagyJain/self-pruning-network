
import numpy as np


def magnitude_scores(weight):
    """Small absolute weights receive lower importance scores."""
    return np.abs(weight)


def taylor_scores(weight, gradient):
    """First-order Taylor importance: absolute weight times gradient."""
    if weight.shape != gradient.shape:
        raise ValueError("Weight and gradient shapes must match")
    return np.abs(weight * gradient)


def cubic_sparsity(step, target_sparsity, warmup_step, final_step):
    """Gradually increase sparsity using a cubic schedule."""
    if not 0.0 <= target_sparsity < 1.0:
        raise ValueError("target_sparsity must be in [0, 1)")
    if final_step <= warmup_step:
        raise ValueError("final_step must be greater than warmup_step")

    if step <= warmup_step:
        return 0.0
    if step >= final_step:
        return float(target_sparsity)

    progress = (step - warmup_step) / (final_step - warmup_step)
    return float(target_sparsity * (1.0 - (1.0 - progress) ** 3))


def global_prune(weights_and_masks, scores, target_sparsity):
    """
    Update masks globally to the requested sparsity.

    weights_and_masks: list of (weight_array, current_mask_array)
    scores: list of importance arrays, one per weight array

    Returns the new masks and achieved global sparsity.
    """
    if not 0.0 <= target_sparsity < 1.0:
        raise ValueError("target_sparsity must be in [0, 1)")
    if len(weights_and_masks) != len(scores):
        raise ValueError("Provide one score array per weight array")

    shapes = []
    active_scores = []

    for (_, mask), score in zip(weights_and_masks, scores):
        if mask.shape != score.shape:
            raise ValueError("Score and mask shapes must match")
        shapes.append(mask.shape)
        active_scores.append(np.asarray(score)[mask.astype(bool)].ravel())

    all_scores = (
        np.concatenate(active_scores)
        if active_scores
        else np.array([], dtype=float)
    )
    total = sum(mask.size for _, mask in weights_and_masks)

    if total == 0:
        return [mask.copy() for _, mask in weights_and_masks], 0.0

    # Never revive already-pruned connections.
    active_count = all_scores.size
    desired_active = int(round(total * (1.0 - target_sparsity)))
    keep_count = min(active_count, desired_active)

    # Stable sorting makes tie handling deterministic.
    keep_indices = np.argsort(all_scores, kind="stable")[-keep_count:] if keep_count else np.array([], dtype=int)
    keep_flat = np.zeros(active_count, dtype=bool)
    keep_flat[keep_indices] = True

    new_masks = []
    offset = 0

    for (_, old_mask), shape in zip(weights_and_masks, shapes):
        old_active = old_mask.astype(bool).ravel()
        layer_keep = np.zeros(old_active.size, dtype=bool)
        count = int(old_active.sum())

        local_keep = keep_flat[offset:offset + count]
        layer_keep[np.flatnonzero(old_active)] = local_keep
        offset += count

        new_masks.append(layer_keep.reshape(shape).astype(float))

    new_active = sum(int(mask.sum()) for mask in new_masks)
    achieved_sparsity = 1.0 - new_active / total
    return new_masks, achieved_sparsity


def layerwise_prune(weights_and_masks, scores, target_sparsity):
    """Prune each layer independently to the requested per-layer sparsity.

    Previously masked weights stay masked. This makes layer-wise pruning useful
    as a controlled comparison with global_prune, which ranks weights across all
    layers in one shared pool.
    """
    if not 0.0 <= target_sparsity < 1.0:
        raise ValueError("target_sparsity must be in [0, 1)")
    if len(weights_and_masks) != len(scores):
        raise ValueError("Provide one score array per weight array")

    new_masks = []
    total_weights = 0
    active_weights = 0

    for (_, old_mask), score in zip(weights_and_masks, scores):
        score = np.asarray(score)
        old_active = np.asarray(old_mask).astype(bool)
        if old_active.shape != score.shape:
            raise ValueError("Score and mask shapes must match")

        flat_active = old_active.ravel()
        flat_scores = score.ravel()
        active_positions = np.flatnonzero(flat_active)
        active_scores = flat_scores[active_positions]
        desired_active = int(round(old_active.size * (1.0 - target_sparsity)))
        keep_count = min(active_positions.size, desired_active)

        keep_active_indices = (
            np.argsort(active_scores, kind="stable")[-keep_count:]
            if keep_count
            else np.array([], dtype=int)
        )
        mask_flat = np.zeros(old_active.size, dtype=bool)
        if keep_count:
            mask_flat[active_positions[keep_active_indices]] = True

        new_mask = mask_flat.reshape(old_active.shape).astype(float)
        new_masks.append(new_mask)
        total_weights += old_active.size
        active_weights += int(new_mask.sum())

    achieved_sparsity = 1.0 - active_weights / total_weights if total_weights else 0.0
    return new_masks, float(achieved_sparsity)

