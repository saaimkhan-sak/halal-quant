from pydantic import BaseModel

class SPShariahRules(BaseModel):
    # Core S&P Shariah thresholds / meta
    max_leverage: float = 0.33           # Debt / 36m average Market Value of Equity < 33%
    buffer: float = 0.02                 # ±2% buffer band for upgrades/downgrades
    consecutive_required: int = 3        # Needs 3 consecutive monthly observations to flip status
    max_impure_rev_ratio: float = 0.05   # Non-permissible revenue (incl. interest) < 5%
    review_frequency: str = "monthly"    # Monday after 3rd Friday (for your scheduling)

    # Prohibited involvement flags (Boolean columns in your CSV)
    banned_categories: tuple = (
        "alcohol","pork","gambling","pornography","tobacco_vaping",
        "conventional_finance","recreational_cannabis",
        "impermissible_advertising","impermissible_media","deferred_gold_silver"
    )

R = SPShariahRules()