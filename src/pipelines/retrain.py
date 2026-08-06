import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import ARTIFACTS_DIR, PROMOTION_TOLERANCE, RETRAIN_COOLDOWN_HOURS
from src.models.train import train_model


def _last_retrain_time():
    meta_path = ARTIFACTS_DIR / "retrain_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return datetime.fromisoformat(json.load(f)["last_retrain"])
    return None


def _save_retrain_meta(status, metrics=None):
    meta = {
        "last_retrain": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    if metrics:
        meta["metrics"] = metrics
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "retrain_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def retrain_pipeline(force=False):
    if not force:
        last = _last_retrain_time()
        if last:
            now = datetime.now(timezone.utc)
            last_aware = last if last.tzinfo else last.replace(tzinfo=timezone.utc)
            if now - last_aware < timedelta(hours=RETRAIN_COOLDOWN_HOURS):
                return {"status": "skipped", "reason": "cooldown_active"}

    current_metrics_path = ARTIFACTS_DIR / "metrics.json"
    old_smape = None
    if current_metrics_path.exists():
        with open(current_metrics_path) as f:
            old_smape = json.load(f)["smape"]

    candidate_dir = ARTIFACTS_DIR / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    try:
        _, candidate_metrics = train_model(output_dir=candidate_dir)
    except Exception as e:
        shutil.rmtree(candidate_dir, ignore_errors=True)
        return {"status": "failed", "error": str(e)}

    new_smape = candidate_metrics["smape"]

    if old_smape is not None and new_smape > old_smape * PROMOTION_TOLERANCE:
        _save_retrain_meta("rejected", candidate_metrics)
        shutil.rmtree(candidate_dir)
        return {
            "status": "rejected",
            "old_smape": old_smape,
            "new_smape": new_smape,
            "threshold": round(old_smape * PROMOTION_TOLERANCE, 4),
        }

    for f in candidate_dir.iterdir():
        shutil.copy2(f, ARTIFACTS_DIR / f.name)
    shutil.rmtree(candidate_dir)

    _save_retrain_meta("promoted", candidate_metrics)
    return {
        "status": "promoted",
        "old_smape": old_smape,
        "new_smape": new_smape,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = retrain_pipeline(force=args.force)
    print(json.dumps(result, indent=2))
