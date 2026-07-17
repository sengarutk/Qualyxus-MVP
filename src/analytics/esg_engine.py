from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config.config import Config


class ESGEngine:
    """Read-only ESG and ROI analytics using vectorized dataframe math."""

    def __init__(self, config: Config | None = None, line_id: str | None = None):
        self.config = config or Config.load()
        self.line_id = line_id

    def _params(self) -> dict[str, Any]:
        base = dict(self.config.get("esg_params", default={}) or {})
        line_overrides = (base.get("line_overrides", {}) or {}).get(self.line_id or "", {})
        merged = {**base, **line_overrides}
        return {
            "material_density_kg_per_unit": float(merged.get("material_density_kg_per_unit", 0.5) or 0.5),
            "unit_cost_usd": float(merged.get("unit_cost_usd", 12.50) or 12.50),
            "carbon_coefficient_kg_co2_per_kg": float(
                merged.get("carbon_coefficient_kg_co2_per_kg", 2.5) or 2.5
            ),
            "currency_symbol": str(merged.get("currency_symbol", "$")),
        }

    def get_impact_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        params = self._params()
        try:
            if df is None or df.empty:
                return {
                    "scrap_prevented_kg": 0.0,
                    "revenue_recovered_usd": 0.0,
                    "co2_emissions_offset_kg": 0.0,
                    "currency_symbol": params["currency_symbol"],
                    "units_count": 0,
                }

            units_count = int(np.nan_to_num(float(df.shape[0]), nan=0.0))
            mass = float(np.nan_to_num(units_count * params["material_density_kg_per_unit"], nan=0.0))
            revenue = float(np.nan_to_num(mass * params["unit_cost_usd"], nan=0.0))
            carbon = float(np.nan_to_num(mass * params["carbon_coefficient_kg_co2_per_kg"], nan=0.0))

            return {
                "scrap_prevented_kg": round(mass, 3),
                "revenue_recovered_usd": round(revenue, 2),
                "co2_emissions_offset_kg": round(carbon, 3),
                "currency_symbol": params["currency_symbol"],
                "units_count": units_count,
            }
        except Exception:
            return {
                "scrap_prevented_kg": 0.0,
                "revenue_recovered_usd": 0.0,
                "co2_emissions_offset_kg": 0.0,
                "currency_symbol": params["currency_symbol"],
                "units_count": 0,
            }