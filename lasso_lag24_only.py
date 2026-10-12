"""Run Lasso with original calendar features plus each target's 24-month lag."""

from lasso_lag24 import main


if __name__ == "__main__":
    main(include_holiday_features=False)
