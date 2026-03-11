"""Shared fixtures for insurance-conformal-fraud tests.

Generates synthetic insurance claim data for testing. No real claim data
is used — all distributions are illustrative.

Genuine claims: approximately multivariate normal with moderate feature
correlations. Fraudulent claims: shifted mean + higher variance, mimicking
the pattern of inflated repair costs and unusual claim timings.
"""

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest


N_FEATURES = 6
N_TRAIN = 300
N_CAL = 150
N_TEST_GENUINE = 100
N_TEST_FRAUD = 20


def _make_genuine(n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate genuine claim features."""
    return rng.multivariate_normal(
        mean=np.zeros(N_FEATURES),
        cov=np.eye(N_FEATURES) * 0.8 + 0.2 * np.ones((N_FEATURES, N_FEATURES)),
        size=n,
    )


def _make_fraud(n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate fraudulent claim features (shifted + higher variance)."""
    return rng.multivariate_normal(
        mean=np.ones(N_FEATURES) * 2.5,
        cov=np.eye(N_FEATURES) * 2.0,
        size=n,
    )


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def X_train(rng) -> np.ndarray:
    return _make_genuine(N_TRAIN, rng)


@pytest.fixture(scope="session")
def X_cal(rng) -> np.ndarray:
    return _make_genuine(N_CAL, rng)


@pytest.fixture(scope="session")
def X_test_genuine(rng) -> np.ndarray:
    return _make_genuine(N_TEST_GENUINE, rng)


@pytest.fixture(scope="session")
def X_test_fraud(rng) -> np.ndarray:
    return _make_fraud(N_TEST_FRAUD, rng)


@pytest.fixture(scope="session")
def X_test_mixed(X_test_genuine, X_test_fraud) -> tuple[np.ndarray, np.ndarray]:
    """Combined test set with ground-truth labels (1=fraud)."""
    X = np.vstack([X_test_genuine, X_test_fraud])
    y = np.array([0] * len(X_test_genuine) + [1] * len(X_test_fraud))
    return X, y


@pytest.fixture(scope="session")
def fitted_scorer(X_train, X_cal):
    """A fitted+calibrated ConformalFraudScorer."""
    from insurance_conformal_fraud import ConformalFraudScorer
    scorer = ConformalFraudScorer(IsolationForest(n_estimators=50, random_state=42))
    scorer.fit(X_train)
    scorer.calibrate(X_cal)
    return scorer


@pytest.fixture(scope="session")
def strata_labels(rng) -> dict:
    """Claim type labels for Mondrian tests."""
    claim_types = ["TPBI", "AD", "THEFT"]
    n_train = 300
    n_cal = 150
    n_test = 120

    rng2 = np.random.default_rng(99)

    train_strata = rng2.choice(claim_types, size=n_train, p=[0.4, 0.4, 0.2])
    cal_strata = rng2.choice(claim_types, size=n_cal, p=[0.4, 0.4, 0.2])
    test_strata = rng2.choice(claim_types, size=n_test, p=[0.4, 0.4, 0.2])

    return {
        "train": train_strata,
        "cal": cal_strata,
        "test": test_strata,
    }
