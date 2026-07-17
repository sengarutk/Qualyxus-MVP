from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.database.logger import emergency_dump_event
from src.logging.metadata_contract import compute_audit_hash, to_metadata_dict


@dataclass
class LogHealthState:
    consecutive_failures: int = 0
    queue_growth_streak: int = 0
    last_queue_size: int = 0
    last_csv_size: int = 0
    last_success_ts: float = 0.0


class LogHealthMonitor:
    def __init__(self, retry_attempts: int = 3, retry_delay: float = 0.2, queue_growth_limit: int = 8):
        self.retry_attempts = max(1, retry_attempts)
        self.retry_delay = retry_delay
        self.queue_growth_limit = queue_growth_limit
        self._state: dict[str, LogHealthState] = {}

    def _get(self, camera_id: str) -> LogHealthState:
        if camera_id not in self._state:
            self._state[camera_id] = LogHealthState()
        return self._state[camera_id]

    def update_queue_health(self, camera_id: str, queue_size: int):
        st = self._get(camera_id)
        if queue_size > st.last_queue_size:
            st.queue_growth_streak += 1
        else:
            st.queue_growth_streak = 0
        st.last_queue_size = queue_size

        if st.queue_growth_streak >= self.queue_growth_limit:
            print(f"[QUALYXUS] ⚠️ Logger queue growth alert for {camera_id}: streak={st.queue_growth_streak}")

    def log_event_with_retry(self, camera_id: str, logger, event, csv_path: str) -> bool:
        st = self._get(camera_id)

        for attempt in range(1, self.retry_attempts + 1):
            try:
                if os.getenv("SIMULATE_DB_LOCK") == "1":
                    raise RuntimeError("Simulated DB lock")

                before_size = Path(csv_path).stat().st_size if Path(csv_path).exists() else 0
                logger.log_event(event)
                after_size = Path(csv_path).stat().st_size if Path(csv_path).exists() else 0

                st.consecutive_failures = 0
                st.last_success_ts = time.time()

                if after_size <= before_size:
                    print(f"[QUALYXUS] ⚠️ CSV export drift detected for {camera_id}: file size did not grow")
                st.last_csv_size = after_size
                return True
            except Exception as e:
                st.consecutive_failures += 1
                print(
                    f"[QUALYXUS] ⚠️ WRITE_FAIL: Log write failed for {camera_id} attempt {attempt}/{self.retry_attempts}: {e}"
                )
                time.sleep(self.retry_delay)

        # Fallback: log full canonical metadata if all write attempts fail
        print(f"[QUALYXUS] 🚨 WRITE_FAIL: All write attempts failed for {camera_id}, falling back to metadata-only logging")
        try:
            metadata_event = to_metadata_dict(event)
            metadata_event["audit_hash"] = metadata_event.get("audit_hash") or compute_audit_hash(metadata_event)
            metadata_event["write_failed"] = True
            metadata_event["failure_reason"] = "disk_write_failure"
            
            # Try to log to a fallback metadata file
            fallback_path = Path(csv_path).parent / f"{Path(csv_path).stem}_metadata{Path(csv_path).suffix}"
            with open(fallback_path, "a", encoding="utf-8") as f:
                import csv
                writer = csv.DictWriter(f, fieldnames=metadata_event.keys())
                if fallback_path.stat().st_size == 0:
                    writer.writeheader()
                writer.writerow(metadata_event)

            # Emergency black-box dump (JSONL) for forensics parity.
            emergency_dump_event(metadata_event)
            
            print(f"[QUALYXUS] ✅ Metadata-only logging succeeded for {camera_id}")
            return True
        except Exception as fallback_e:
            print(f"[QUALYXUS] 🚨 WRITE_FAIL: Metadata fallback also failed for {camera_id}: {fallback_e}")
            return False

    def failure_count(self, camera_id: str) -> int:
        return self._get(camera_id).consecutive_failures
