import joblib
import numpy as np
import pytest

# Load model
model = joblib.load("credit_risk_model.pkl")

# Categorical encodings (sorted alphabetically, matching LabelEncoder behavior)
HOME_OWNERSHIP = ["MORTGAGE", "OTHER", "OWN", "RENT"]
LOAN_INTENT = ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
LOAN_GRADE = ["A", "B", "C", "D", "E", "F", "G"]
DEFAULT_ON_FILE = ["N", "Y"]


def encode(cat_list, value):
    return cat_list.index(value)


def make_features(age, income, home, emp_length, intent, grade, loan_amnt, int_rate, pct_income, default_hist, cred_hist):
    return np.array([[
        age, income,
        encode(HOME_OWNERSHIP, home),
        emp_length,
        encode(LOAN_INTENT, intent),
        encode(LOAN_GRADE, grade),
        loan_amnt, int_rate, pct_income,
        encode(DEFAULT_ON_FILE, default_hist),
        cred_hist
    ]])


# --- Test Cases ---

class TestLowRiskScenarios:
    """Scenarios that should predict loan approval (status=0)"""

    def test_high_income_low_loan(self):
        """High income, small loan, good grade, no default history"""
        X = make_features(35, 120000, "OWN", 10.0, "PERSONAL", "A", 5000, 6.0, 0.04, "N", 15)
        pred = model.predict(X)[0]
        assert pred == 0, f"Expected low risk (0), got {pred}"

    def test_stable_employment_mortgage(self):
        """Long employment, mortgage owner, education loan"""
        X = make_features(40, 85000, "MORTGAGE", 15.0, "EDUCATION", "B", 10000, 8.5, 0.12, "N", 20)
        pred = model.predict(X)[0]
        assert pred == 0, f"Expected low risk (0), got {pred}"

    def test_moderate_income_small_loan(self):
        """Average income, very small loan relative to income"""
        X = make_features(30, 60000, "RENT", 5.0, "MEDICAL", "B", 3000, 9.0, 0.05, "N", 8)
        pred = model.predict(X)[0]
        assert pred == 0, f"Expected low risk (0), got {pred}"

    def test_older_applicant_good_history(self):
        """Older applicant with long credit history"""
        X = make_features(55, 95000, "OWN", 25.0, "HOMEIMPROVEMENT", "A", 15000, 5.5, 0.16, "N", 30)
        pred = model.predict(X)[0]
        assert pred == 0, f"Expected low risk (0), got {pred}"


class TestHighRiskScenarios:
    """Scenarios that should predict default (status=1)"""

    def test_low_income_high_loan(self):
        """Low income, large loan, bad grade, previous default"""
        X = make_features(22, 15000, "RENT", 1.0, "PERSONAL", "F", 35000, 22.0, 0.70, "Y", 2)
        pred = model.predict(X)[0]
        assert pred == 1, f"Expected high risk (1), got {pred}"

    def test_young_no_history_high_rate(self):
        """Young, short credit history, high interest rate"""
        X = make_features(21, 20000, "RENT", 0.5, "VENTURE", "E", 25000, 20.0, 0.60, "Y", 1)
        pred = model.predict(X)[0]
        assert pred == 1, f"Expected high risk (1), got {pred}"

    def test_high_debt_to_income(self):
        """Very high loan percentage of income"""
        X = make_features(25, 25000, "OTHER", 2.0, "DEBTCONSOLIDATION", "D", 20000, 18.0, 0.80, "Y", 3)
        pred = model.predict(X)[0]
        assert pred == 1, f"Expected high risk (1), got {pred}"

    def test_worst_grade_max_rate(self):
        """Worst possible loan grade with high interest"""
        X = make_features(23, 18000, "RENT", 1.0, "PERSONAL", "G", 30000, 25.0, 0.65, "Y", 2)
        pred = model.predict(X)[0]
        assert pred == 1, f"Expected high risk (1), got {pred}"


class TestEdgeCases:
    """Boundary and edge case scenarios"""

    def test_minimum_age(self):
        """Youngest possible applicant"""
        X = make_features(18, 12000, "RENT", 0.0, "EDUCATION", "C", 5000, 12.0, 0.42, "N", 1)
        pred = model.predict(X)[0]
        assert pred in [0, 1]

    def test_very_high_income(self):
        """Extremely high income applicant"""
        X = make_features(45, 500000, "OWN", 20.0, "VENTURE", "A", 50000, 5.0, 0.10, "N", 25)
        pred = model.predict(X)[0]
        assert pred == 0, f"Expected low risk (0), got {pred}"

    def test_zero_employment_length(self):
        """No employment history"""
        X = make_features(20, 10000, "RENT", 0.0, "PERSONAL", "D", 10000, 15.0, 0.50, "N", 1)
        pred = model.predict(X)[0]
        assert pred in [0, 1]

    def test_max_credit_history(self):
        """Very long credit history"""
        X = make_features(65, 70000, "MORTGAGE", 30.0, "HOMEIMPROVEMENT", "B", 20000, 7.5, 0.29, "N", 40)
        pred = model.predict(X)[0]
        assert pred == 0, f"Expected low risk (0), got {pred}"


class TestAllCategoricalValues:
    """Ensure model handles all categorical value encodings"""

    @pytest.mark.parametrize("home", HOME_OWNERSHIP)
    def test_all_home_ownership(self, home):
        X = make_features(30, 50000, home, 5.0, "PERSONAL", "C", 10000, 12.0, 0.20, "N", 5)
        pred = model.predict(X)[0]
        assert pred in [0, 1]

    @pytest.mark.parametrize("intent", LOAN_INTENT)
    def test_all_loan_intents(self, intent):
        X = make_features(30, 50000, "RENT", 5.0, intent, "C", 10000, 12.0, 0.20, "N", 5)
        pred = model.predict(X)[0]
        assert pred in [0, 1]

    @pytest.mark.parametrize("grade", LOAN_GRADE)
    def test_all_loan_grades(self, grade):
        X = make_features(30, 50000, "RENT", 5.0, "PERSONAL", grade, 10000, 12.0, 0.20, "N", 5)
        pred = model.predict(X)[0]
        assert pred in [0, 1]

    @pytest.mark.parametrize("default", DEFAULT_ON_FILE)
    def test_all_default_values(self, default):
        X = make_features(30, 50000, "RENT", 5.0, "PERSONAL", "C", 10000, 12.0, 0.20, default, 5)
        pred = model.predict(X)[0]
        assert pred in [0, 1]


class TestModelOutput:
    """Verify model output format and probabilities"""

    def test_prediction_is_binary(self):
        X = make_features(30, 50000, "RENT", 5.0, "PERSONAL", "C", 10000, 12.0, 0.20, "N", 5)
        pred = model.predict(X)[0]
        assert pred in [0, 1]

    def test_probabilities_sum_to_one(self):
        X = make_features(30, 50000, "RENT", 5.0, "PERSONAL", "C", 10000, 12.0, 0.20, "N", 5)
        proba = model.predict_proba(X)[0]
        assert abs(proba.sum() - 1.0) < 1e-6

    def test_probability_range(self):
        X = make_features(30, 50000, "RENT", 5.0, "PERSONAL", "C", 10000, 12.0, 0.20, "N", 5)
        proba = model.predict_proba(X)[0]
        assert all(0 <= p <= 1 for p in proba)
