"""Small Poisson count models; all learned transformations fit at origin."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from .features import eligible_training

HISTORY_COLUMNS = ("history_age_days", "kyiv_lag1", "kyiv_lag7", "kyiv_mean3", "kyiv_mean7",
                   "kyiv_mean28", "neighbors_lag1", "neighbors_mean7", "weekday_sin", "weekday_cos")
NEWS_COLUMNS = ("news_titles3", "news_tone3", "news_conflict_fraction3")
TEXT_CAP = 50


@dataclass(frozen=True)
class ModelConfig:
    with_news: bool
    alpha: float
    train_window: int | None
    level_offset: bool = False

    @property
    def name(self):
        family = "history_news_poisson" if self.with_news else "history_poisson"
        level = "recent7" if self.level_offset else "direct"
        return f"{family}__{level}__alpha{self.alpha:g}__window{self.train_window or 'all'}"

    def to_dict(self):
        return {"name": self.name, **asdict(self)}


def fit_predict_poisson(train, row, config: ModelConfig):
    if train.empty or len(row) != 1:
        raise ValueError("Nonempty training and exactly one forecast row required")
    origin = pd.Timestamp(row.iloc[0].origin)
    # The estimator enforces timing itself, not just its callers. This guards
    # accidental full-frame fits and excludes future/delayed outcomes.
    train = eligible_training(train, origin).sort_values("target_day")
    if config.train_window is not None:
        train = train.tail(config.train_window)
    if len(train) < 50:
        raise ValueError("Fewer than 50 available training labels")
    numeric_columns = list(HISTORY_COLUMNS + (NEWS_COLUMNS if config.with_news else ()))
    # Drop zero-variance inputs using ONLY this origin's eligible training set.
    active = [c for c in numeric_columns if train[c].nunique() > 1]
    scaler = StandardScaler()
    if active:
        x = csr_matrix(scaler.fit_transform(train[active]))
        z = csr_matrix(scaler.transform(row[active]))
    else:
        x, z = csr_matrix((len(train), 0)), csr_matrix((1, 0))
    vocabulary = []
    if config.with_news:
        # At most one text coordinate per three training observations, capped
        # at 50. Small rolling fits therefore use <50 terms, not 50 by default.
        cap = min(TEXT_CAP, len(train) // 3)
        vectorizer = TfidfVectorizer(max_features=cap, min_df=3, max_df=.9,
                                     ngram_range=(1, 1), stop_words="english",
                                     sublinear_tf=True)
        try:
            texts = vectorizer.fit_transform(train.titles.fillna(""))
            future_text = vectorizer.transform(row.titles.fillna(""))
            x, z = hstack([x, texts]).tocsr(), hstack([z, future_text]).tocsr()
            vocabulary = vectorizer.get_feature_names_out().tolist()
        except ValueError as exc:
            if "empty vocabulary" not in str(exc) and "no terms remain" not in str(exc).lower():
                raise
    # An all-constant synthetic/history set still supports an intercept model.
    if not x.shape[1]:
        x, z = csr_matrix(np.zeros((len(train), 1))), csr_matrix(np.zeros((1, 1)))
    model = PoissonRegressor(alpha=config.alpha, solver="lbfgs", max_iter=2000, tol=1e-7)
    # Weighted-rate identity: fitting y/e with weight e is exactly the count
    # Poisson likelihood for mean e*exp(X beta), up to parameter-independent
    # constants. e is computed from earlier available starts, not future truth.
    # A fixed half-count in a seven-day window permits a genuinely quiet history.
    exposure = train.kyiv_mean7.to_numpy() + .5 / 7 if config.level_offset else np.ones(len(train))
    forecast_exposure = float(row.iloc[0].kyiv_mean7 + .5 / 7) if config.level_offset else 1.
    if np.any(exposure <= 0) or forecast_exposure <= 0:
        raise ValueError("Invalid recent-level exposure")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, train.y_train_vintage.to_numpy() / exposure, sample_weight=exposure)
    prediction = float(forecast_exposure * model.predict(z)[0])
    # The log link gives exp(X beta + intercept), without clipping a linear
    # prediction or converting an expected count to a rounded integer.
    if not np.isfinite(prediction) or prediction <= 0:
        raise ValueError("Invalid Poisson expected count; fail rather than clip")
    audit = {"training_rows": len(train), "max_train_origin": str(train.origin.max()),
             "max_train_target_end": str(train.target_end.max()),
             "max_train_label_available_at": str(train.label_available_at.max()),
             "numeric_features": active, "numeric_feature_count": len(active),
             "text_feature_count": len(vocabulary), "vocabulary": vocabulary,
             "total_feature_count": x.shape[1], "iterations": int(model.n_iter_),
             "forecast_recent_level_exposure": forecast_exposure,
             "model": config.to_dict()}
    return prediction, audit


def baseline_predictions(frame, row):
    train = eligible_training(frame, row.iloc[0].origin)
    same = train[train.target_day.dt.dayofweek == row.iloc[0].target_day.dayofweek].tail(8)
    return {"rolling28": float(row.iloc[0].kyiv_mean28),
            "rolling7": float(row.iloc[0].kyiv_mean7),
            "weekday8": float(same.y_train_vintage.mean()) if len(same) else float(row.iloc[0].kyiv_mean28)}
