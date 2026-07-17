from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger(__name__)


IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
EVENT_LOG_GLOBS = ("*.csv", "*.db", "*.sqlite", "*.sqlite3", "*.json", "*.log", "*.txt")


@dataclass(frozen=True)
class RetentionPolicy:
    """Retention settings for a single storage target."""

    name: str
    source_dir: str
    retention_days: int
    archive_enabled: bool = False
    archive_dir: str | None = None
    include_globs: tuple[str, ...] = ("*",)


@dataclass(frozen=True)
class RetentionConfig:
    """Top-level retention configuration for the three required categories."""

    raw_images: RetentionPolicy
    defect_images: RetentionPolicy
    event_logs: RetentionPolicy

    def policies(self) -> tuple[RetentionPolicy, RetentionPolicy, RetentionPolicy]:
        return (self.raw_images, self.defect_images, self.event_logs)


@dataclass
class PolicyCleanupResult:
    policy_name: str
    scanned_files: int = 0
    expired_files: int = 0
    archived_files: int = 0
    deleted_files: int = 0
    skipped_files: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class RetentionRunResult:
    started_at_epoch: float
    ended_at_epoch: float
    dry_run: bool
    per_policy: list[PolicyCleanupResult]

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.ended_at_epoch - self.started_at_epoch)


class DataRetentionManager:
    """
    Cleans files older than policy thresholds and optionally archives them first.

    Safety guarantees:
    - never deletes directories
    - never follows symlink targets for destructive operations
    - only acts on files rooted under each policy's source directory
    - archives before deleting when archive mode is enabled
    """

    def __init__(self, config: RetentionConfig):
        self.config = config
        self._validate_config()

    @classmethod
    def default(cls) -> "DataRetentionManager":
        """Reasonable defaults for this project layout."""
        cfg = RetentionConfig(
            raw_images=RetentionPolicy(
                name="raw_images",
                source_dir="data/live/raw",
                retention_days=30,
                archive_enabled=False,
                include_globs=IMAGE_GLOBS,
            ),
            defect_images=RetentionPolicy(
                name="defect_images",
                source_dir="data/live/defects",
                retention_days=90,
                archive_enabled=True,
                archive_dir="data/archive/defects",
                include_globs=IMAGE_GLOBS,
            ),
            event_logs=RetentionPolicy(
                name="event_logs",
                source_dir="data/events",
                retention_days=180,
                archive_enabled=True,
                archive_dir="data/archive/events",
                include_globs=EVENT_LOG_GLOBS,
            ),
        )
        return cls(cfg)

    @classmethod
    def from_dict(cls, raw: dict) -> "DataRetentionManager":
        """
        Build manager from dictionary configuration.

        Expected shape:
        {
          "retention": {
            "raw_images": {...},
            "defect_images": {...},
            "event_logs": {...}
          }
        }
        """
        retention = raw.get("retention", raw)

        def _policy(name: str, default_globs: tuple[str, ...]) -> RetentionPolicy:
            data = retention.get(name, {})
            return RetentionPolicy(
                name=name,
                source_dir=str(data.get("source_dir", "")),
                retention_days=int(data.get("retention_days", 0)),
                archive_enabled=bool(data.get("archive_enabled", False)),
                archive_dir=data.get("archive_dir"),
                include_globs=tuple(data.get("include_globs", default_globs)),
            )

        cfg = RetentionConfig(
            raw_images=_policy("raw_images", IMAGE_GLOBS),
            defect_images=_policy("defect_images", IMAGE_GLOBS),
            event_logs=_policy("event_logs", EVENT_LOG_GLOBS),
        )
        return cls(cfg)

    def run_cleanup(self, dry_run: bool = False, now_epoch: float | None = None) -> RetentionRunResult:
        started = time.time()
        now_epoch = now_epoch if now_epoch is not None else started

        per_policy: list[PolicyCleanupResult] = []
        for policy in self.config.policies():
            per_policy.append(self._cleanup_policy(policy=policy, now_epoch=now_epoch, dry_run=dry_run))

        return RetentionRunResult(
            started_at_epoch=started,
            ended_at_epoch=time.time(),
            dry_run=dry_run,
            per_policy=per_policy,
        )

    def _validate_config(self):
        names = set()
        for policy in self.config.policies():
            if policy.name in names:
                raise ValueError(f"Duplicate policy name: {policy.name}")
            names.add(policy.name)

            if not policy.source_dir:
                raise ValueError(f"Policy '{policy.name}' requires non-empty source_dir")
            if policy.retention_days < 0:
                raise ValueError(f"Policy '{policy.name}' retention_days cannot be negative")

            if policy.archive_enabled and not policy.archive_dir:
                raise ValueError(f"Policy '{policy.name}' has archive_enabled=True but no archive_dir")

    def _cleanup_policy(self, policy: RetentionPolicy, now_epoch: float, dry_run: bool) -> PolicyCleanupResult:
        result = PolicyCleanupResult(policy_name=policy.name)

        source_root = Path(policy.source_dir)
        if not source_root.exists():
            LOGGER.info("Retention source does not exist for policy '%s': %s", policy.name, source_root)
            return result

        cutoff_epoch = now_epoch - float(policy.retention_days) * 24 * 60 * 60
        for file_path in self._iter_candidate_files(source_root=source_root, globs=policy.include_globs):
            result.scanned_files += 1

            if not self._is_safe_file_candidate(source_root=source_root, file_path=file_path):
                result.skipped_files += 1
                continue

            try:
                modified_epoch = file_path.stat().st_mtime
            except OSError as exc:
                result.errors.append(f"{file_path}: stat failed ({exc})")
                continue

            if modified_epoch > cutoff_epoch:
                continue

            result.expired_files += 1

            try:
                if policy.archive_enabled:
                    if self._archive_file(policy=policy, source_root=source_root, file_path=file_path, dry_run=dry_run):
                        result.archived_files += 1

                if dry_run:
                    LOGGER.info("[dry-run] delete expired file: %s", file_path)
                    result.deleted_files += 1
                else:
                    file_path.unlink(missing_ok=True)
                    result.deleted_files += 1
            except Exception as exc:  # broad catch to avoid aborting entire cleanup run
                result.errors.append(f"{file_path}: cleanup failed ({exc})")

        return result

    def _iter_candidate_files(self, source_root: Path, globs: Iterable[str]) -> Iterable[Path]:
        seen: set[Path] = set()
        for pattern in globs:
            for file_path in source_root.rglob(pattern):
                if file_path in seen:
                    continue
                seen.add(file_path)
                if file_path.is_file():
                    yield file_path

    def _is_safe_file_candidate(self, source_root: Path, file_path: Path) -> bool:
        """
        Safety filter before destructive operations.
        """
        if file_path.is_symlink():
            LOGGER.warning("Skipping symlink during retention cleanup: %s", file_path)
            return False

        try:
            source_resolved = source_root.resolve()
            candidate_resolved = file_path.resolve()
            candidate_resolved.relative_to(source_resolved)
        except Exception:
            LOGGER.warning("Skipping unsafe candidate outside source root: %s", file_path)
            return False

        return True

    def _archive_file(self, policy: RetentionPolicy, source_root: Path, file_path: Path, dry_run: bool) -> bool:
        archive_root = Path(policy.archive_dir or "")

        # Keep source subfolder structure inside archive root.
        relative_path = file_path.resolve().relative_to(source_root.resolve())
        archive_target = archive_root / relative_path

        if dry_run:
            LOGGER.info("[dry-run] archive %s -> %s", file_path, archive_target)
            return True

        archive_target.parent.mkdir(parents=True, exist_ok=True)

        # copy2 preserves metadata; we only delete original after successful copy.
        shutil.copy2(file_path, archive_target)
        return True

    def get_retention_status(self) -> dict:
        """Get current retention status metrics."""
        total_files = 0
        total_storage_gb = 0.0
        projected_deletions = 0
        now_epoch = time.time()
        for policy in self.config.policies():
            source_root = Path(policy.source_dir)
            if source_root.exists():
                for file_path in self._iter_candidate_files(source_root=source_root, globs=policy.include_globs):
                    if self._is_safe_file_candidate(source_root=source_root, file_path=file_path):
                        total_files += 1
                        try:
                            size_bytes = file_path.stat().st_size
                            total_storage_gb += size_bytes / (1024**3)
                        except OSError:
                            pass
            result = self._cleanup_policy(policy=policy, now_epoch=now_epoch, dry_run=True)
            projected_deletions += result.deleted_files
        return {
            "total_files": total_files,
            "total_storage_gb": round(total_storage_gb, 2),
            "projected_deletions": projected_deletions,
            "next_cleanup_utc": None,
        }

__all__ = [
    "DataRetentionManager",
    "RetentionConfig",
    "RetentionPolicy",
    "RetentionRunResult",
    "PolicyCleanupResult",
    "IMAGE_GLOBS",
    "EVENT_LOG_GLOBS",
]
