"""
Achille Nazaret

Regression discontinuity analysis.
"""

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def ratio_a_over_b(
    mean_a: float, std_a: float, mean_b: float, std_b: float, confidence: float
) -> tuple[float, float, float]:
    """
    Compute the confidence interval of the ratio of two independent normal distributions a and b.
    It is approximated by the delta method.

    Parameters
    ----------
    mean_a : float
        The mean of the first normal distribution.
    std_a : float
        The standard deviation of the first normal distribution.
    mean_b : float
        The mean of the second normal distribution.
    std_b : float
        The standard deviation of the second normal distribution.
    confidence : float
        The confidence level of the confidence interval. E.g. 0.95 for a 95% confidence interval.

    Returns
    -------
    ratio : float
        The ratio of the means of the two normal distributions.
    low : float
        The lower bound of the confidence interval.
    upper : float
        The upper bound of the confidence interval.
    """
    ratio = mean_a / mean_b
    tq = scipy.stats.norm.ppf(1 - (1 - confidence) / 2)

    s2 = std_a**2 / mean_a**2 + std_b**2 / mean_b**2
    s = abs(ratio) * tq * np.sqrt(s2)
    low, upper = ratio - s, ratio + s

    return ratio, low, upper


def fit_regression_discontinuity(
    data: pd.DataFrame,
    target_hours: list[int],
    target_minutes: list[int],
    nudge_at_delta=11,
    confidence_level=0.99,
    model="linear",
):
    """
    Fit a regression discontinuity model on the standing data, either linear or logistic.
    Compute the absolute and relative causal effect of the nudging, as well as the corresponding
    confidence intervals.

    Parameters
    ----------
    data : pd.DataFrame
        The standing data to fit the model on.
    target_hours : list of int
        The hours to consider for the regression discontinuity.
    target_minutes : list of int
        The minutes to consider for the regression discontinuity.
    nudge_at_delta : int, optional
        The delta at which the nudging is applied, by default 11, which corresponds to 50 minutes of prolonged sitting.
        In more details `-delta` is the index of the last time interval when the user was standing.
        For e.g.:
        - delta=1 corresponds to users standing in the previous time interval.
        - delta=11 means 10 interval of 5 minutes sitting and the eleventh is a standing
          interval, hence 50 minutes of prolonged sitting. This is the default value.
    confidence_level : float, optional
        The confidence level of the confidence intervals, by default 0.99.
    model : str, optional
        The model to use, either "linear" or "logistic", by default "linear".

    Returns
    -------
    dict
        The regression discontinuity model.

    """
    data = data[
        (data["target_bucket"].dt.minute.isin(target_minutes))
        & (data["target_bucket"].dt.hour.isin(target_hours))
    ].copy()
    data_pd = {c: data[c] for c in data.columns}

    data_pd["is_nudged"] = (data["delta"] >= nudge_at_delta).astype(int)
    data = pd.DataFrame(data_pd)

    if model == "linear":
        return _fit_regression_discontinuity_linear(data, nudge_at_delta, confidence_level)
    elif model == "logistic":
        return _fit_regression_discontinuity_logistic(data, nudge_at_delta, confidence_level)
    else:
        raise ValueError(f"Unknown model {model}")


def _fit_regression_discontinuity_linear(
    data: pd.DataFrame, nudge_at_delta, confidence_level=0.99
) -> dict:
    """
    Fit a linear regression model to the data.
    See fit_regression_discontinuity for more details.
    """
    variables = ["delta", "is_nudged"]

    linear_model = smf.glm(
        f"stand ~ " + " + ".join(variables),
        data=data,
        freq_weights=data["count"],
        family=sm.families.Gaussian(),
    ).fit()

    causal_effect = linear_model.params["is_nudged"]
    causal_confint_low = linear_model.conf_int(0.01)[0]["is_nudged"]
    causal_confint_high = linear_model.conf_int(0.01)[1]["is_nudged"]

    standing_rate_treated = linear_model.predict(dict(delta=nudge_at_delta, is_nudged=1))[0]
    standing_rate_untreated = linear_model.predict(dict(delta=nudge_at_delta, is_nudged=0))[0]

    assert abs(standing_rate_treated - standing_rate_untreated - causal_effect) < 1e-5

    relative, relative_low, relative_high = ratio_a_over_b(
        causal_effect,
        linear_model.bse["is_nudged"],
        standing_rate_untreated,
        (linear_model.bse["Intercept"] ** 2 + linear_model.bse["delta"] ** 2 * nudge_at_delta**2)
        ** 0.5,
        confidence_level,
    )

    linear = {
        "causal_effect": causal_effect,
        "causal_confint_low": causal_confint_low,
        "causal_confint_high": causal_confint_high,
        "standing_rate_treated": standing_rate_treated,
        "standing_rate_untreated": standing_rate_untreated,
        "relative": relative,
        "relative_low": relative_low,
        "relative_high": relative_high,
        "pvalue": linear_model.pvalues["is_nudged"],
        "counts": data["count"].sum(),
    }

    return linear


def _fit_regression_discontinuity_logistic(data, nudge_at_delta, confidence_level=0.99) -> dict:
    """
    Fit a logistic regression model to the data.
    See fit_regression_discontinuity for more details.
    """

    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    variables = ["delta", "is_nudged"]

    logistic_model = smf.glm(
        f"stand ~ " + " + ".join(variables),
        data=data,
        freq_weights=data["count"],
        family=sm.families.Binomial(),
    ).fit()

    standing_rate_treated = logistic_model.predict(dict(delta=nudge_at_delta, is_nudged=1))[0]
    standing_rate_untreated = logistic_model.predict(dict(delta=nudge_at_delta, is_nudged=0))[0]

    causal_effect = standing_rate_treated - standing_rate_untreated

    # TODO: implement these confidence intervals if needed, there will be 0 for now
    causal_confint_low = 0 * (
        sigmoid(
            logistic_model.params["Intercept"]
            + nudge_at_delta * logistic_model.params["delta"]
            + logistic_model.conf_int(1 - confidence_level)[0]["is_nudged"]
        )
        - standing_rate_untreated
    )
    causal_confint_high = 0 * (
        sigmoid(
            logistic_model.params["Intercept"]
            + nudge_at_delta * logistic_model.params["delta"]
            + logistic_model.conf_int(1 - confidence_level)[1]["is_nudged"]
        )
        - standing_rate_untreated
    )

    relative = causal_effect / standing_rate_untreated

    tq = scipy.stats.norm.ppf(1 - (1 - confidence_level) / 2)
    var_x = (
        logistic_model.bse["Intercept"] ** 2
        + logistic_model.bse["delta"] ** 2 * nudge_at_delta**2
    )
    var_tau = logistic_model.bse["is_nudged"] ** 2
    v2 = var_tau * (1 - standing_rate_treated) ** 2
    v1 = var_x * (standing_rate_treated - standing_rate_untreated) ** 2
    s = abs(relative + 1) * tq * np.sqrt(v1 + v2)
    relative_low, relative_high = relative - s, relative + s

    logistic = {
        "causal_effect": causal_effect,
        "causal_confint_low": causal_confint_low,
        "causal_confint_high": causal_confint_high,
        "standing_rate_treated": standing_rate_treated,
        "standing_rate_untreated": standing_rate_untreated,
        "relative": relative,
        "relative_low": relative_low,
        "relative_high": relative_high,
        "pvalue": logistic_model.pvalues["is_nudged"],
        "counts": data["count"].sum(),
    }

    return logistic
