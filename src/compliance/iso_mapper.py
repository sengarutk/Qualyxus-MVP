from __future__ import annotations

from typing import Any


class ISOMapper:
    """Maps 6M classifications to ISO 9001:2015 clauses."""

    CLAUSE_MAP: dict[str, str] = {
        "Machine": "Clause 7.1.3 Infrastructure",
        "Man": "Clause 7.2 Competence",
        "Method": "Clause 8.5.1 Control of Production",
        "Material": "Clause 8.4 External Provider Control",
        "Measurement": "Clause 9.1.1 Monitoring & Measurement",
        "Environment": "Clause 7.1.4 Process Environment",
    }

    # Explicit defect-type overrides to satisfy compliance expectations.
    # (These take precedence over 6M heuristic inference.)
    DEFECT_TYPE_CLAUSE_OVERRIDES: list[tuple[str, str]] = [
        # Nonconforming output handling
        ("missing component", "Clause 8.7 Non-conforming Outputs"),
        ("missing", "Clause 8.7 Non-conforming Outputs"),
        # Production control / workmanship
        ("surface scratch", "Clause 8.5.1 Control of Production"),
        ("scratch", "Clause 8.5.1 Control of Production"),

        # User requirement: map "Surface Crack" to ISO 9001:2015 9.1.1 (Monitoring & Measurement)
        # (Normalization uses lowercase contains-match.)
        ("surface crack", "Clause 9.1.1 Monitoring & Measurement"),
        ("crack", "Clause 9.1.1 Monitoring & Measurement"),
    ]

    @classmethod
    def clause_for_category(cls, category: str | None) -> str | None:
        if not category:
            return None
        return cls.CLAUSE_MAP.get(str(category))

    @classmethod
    def infer_category_from_event(cls, event: Any) -> str:
        defect_type = str(getattr(event, "defect_type", "") or "").lower()
        if any(getattr(event, field, None) is not None for field in ("temperature", "vibration", "speed")):
            return "Measurement"
        if getattr(event, "operator_id", None):
            return "Man"
        if getattr(event, "machine_id", None):
            return "Machine"
        if "material" in defect_type or "crack" in defect_type or "corrosion" in defect_type:
            return "Material"
        if getattr(event, "shift_id", None):
            return "Environment"
        return "Method"

    @classmethod
    def clause_for_event(cls, event: Any) -> str | None:
        explicit = getattr(event, "iso_clause", None)
        if explicit:
            return explicit
        defect_type = str(getattr(event, "defect_type", "") or "").lower()
        for needle, clause in cls.DEFECT_TYPE_CLAUSE_OVERRIDES:
            if needle in defect_type:
                return clause
        return cls.clause_for_category(cls.infer_category_from_event(event))