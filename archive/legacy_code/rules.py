from pydantic import BaseModel

class ShariahRules(BaseModel):
    # Common thresholds used by major Shariah index methodologies
    max_debt_to_assets: float = 0.3333
    max_cash_interest_to_assets: float = 0.3333
    max_ar_plus_cash_to_assets: float = 0.50
    max_impure_income_ratio: float = 0.05  # <= 5% impure income

DEFAULT_RULES = ShariahRules()