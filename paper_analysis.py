import numpy as np
import pandas as pd

from utils import load_data, get_n_users_m_days
from regression_discontinuity import fit_regression_discontinuity


def analyze_each_control_group(model: str = "linear") -> pd.DataFrame:
    """
    Analyze each control group to (re)produce the tables of the paper.

    Parameters
    ----------
    model : str, optional (default="linear")
        The model to use for the regression discontinuity. Either "linear" or "logistic".

    Returns
    -------
    results : pd.DataFrame
        The results of the analysis.
    """

    results = dict()

    for control_variable, values in [
        ("All", {"--": [None]}),
        (
            "age_group",
            {
                "18 -- 24": [2],
                "25 -- 34": [3],
                "35 -- 44": [4],
                "45 -- 54": [5],
                "55 -- 64": [6],
                "65 -- 74": [7],
                "75 -- 84": [8],
            },
        ),
        ("biological_sex", {"Male": ["Male"], "Female": ["Female"], "Other": ["Other"]}),
        (
            "day_of_week",
            {
                "Weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "Weekend": ["Saturday", "Sunday"],
            },
        ),
        (
            "energy_group",
            {
                "0 -- 49": [0, 1, 2, 3, 4],
                "50 -- 79": [5, 6, 7],
                "80 -- 99": [8, 9],
                "9##100": [10],
            },
        ),
        (
            "exercise_group",
            {"0 -- 49": [0, 1, 2, 3, 4], "50 -- 79": [5, 6, 7], "80 -- 99": [8, 9], "9##100": [10]},
        ),
        (
            "stand_group",
            {
                "0 -- 49": [0, 1, 2, 3, 4],
                "50 -- 79": [5, 6, 7],
                "80 -- 99": [8, 9],
                "9##100": [10],  # Just for sorting in alphabetical order
            },
        ),
    ]:
        df = load_data("../data", control_variable=control_variable)
        # only keep the delta between 7 and 15 (inclusive), that is, the
        # data with prolonged sitting between 30 and 70 minutes
        df = df[df["delta"].between(7, 15)]
        df = df[df["delta"] != 12]

        for k, v in values.items():
            if control_variable is not None and control_variable != "All":
                data_tmp = df[df[control_variable].isin(v)].copy()
            else:
                data_tmp = df
            data_tmp["stand"] = (data_tmp["stand_minutes"] > 0).astype(int)
            data_tmp = (
                data_tmp.groupby(["delta", "target_bucket", "stand"])["count"].sum().reset_index()
            )

            for target_hours_name, target_hours in [
                ("morning", list(range(6, 12))),
                ("afternoon", list(range(12, 18))),
                ("evening", list(range(18, 24))),
            ]:
                results[(control_variable, k, target_hours_name)] = fit_regression_discontinuity(
                    data_tmp,
                    target_hours=target_hours,
                    target_minutes=[50],
                    nudge_at_delta=11,
                    model=model,
                )

    results = pd.DataFrame(results).T

    results["report_relative"] = results.apply(
        lambda x: f'${np.round(x["relative"]*100,1)} \\%$ '
        f'($99\\%$ CI $({np.round(x["relative_low"]*100,1)}\\%, '
        f'{np.round(x["relative_high"]*100,1)}\\%)$)',
        axis=1,
    )
    results["report_absolute"] = results.apply(
        lambda x: f'${np.round(x["causal_effect"]*100,2)}$ '
        f'($99\\%$ CI $({np.round(x["causal_confint_low"]*100,2)}\\%, '
        f'{np.round(x["causal_confint_high"]*100,2)}\\%)$)',
        axis=1,
    )
    results["table_relative"] = results.apply(
        lambda x: f'\${np.round(x["relative"]*100,1)} ~ \scriptstyle \\pm '
        + str(
            max(
                np.round((x["relative_high"] - x["relative"]) * 100, 1),
                np.round((x["relative"] - x["relative_low"]) * 100, 1),
            )
        )
        + "\$",
        axis=1,
    )
    results["table_absolute"] = results.apply(
        lambda x: f'\${np.round(x["causal_effect"]*100,2)} ~ \scriptstyle \\pm {max(np.round((x["causal_confint_high"] - x["causal_effect"])*100,2), np.round((x["causal_effect"] - x["causal_confint_low"])*100,2))}\$',
        axis=1,
    )
    results.index.names = ["variable", "subgroup", "time_of_day"]

    return results



def analysis_to_latex_table(results: pd.DataFrame, effect: str = "relative", display_notebook=False) -> str:
    """Converts the results of the analysis to a LaTeX table.

    Parameters
    ----------
    results : pd.DataFrame
        The results of the analysis.
    effect : str, optional
        The effect to report. Can be either "relative" or "absolute", by default "relative".
    notebook_display : bool, optional
        Whether to display the table in a notebook, by default False.

    Returns
    -------
    str
        The LaTeX table.
    """
    val = "table_" + effect

    tmp = pd.pivot_table(
        results[val].reset_index(),
        values=val,
        index=["variable", "subgroup"],
        columns="time_of_day",
        aggfunc="first",
    )
    tmp = tmp.reset_index().replace("9##100", "100").set_index(["variable", "subgroup"])
    tmp["$n$ users"] = (
        tmp.reset_index()["subgroup"].apply(get_n_users_m_days, key="user").values
    )
    tmp["$m$ user-days"] = (
        tmp.reset_index()["subgroup"].apply(get_n_users_m_days, key="user-day").values
    )
    tmp = tmp[["$n$ users", "$m$ user-days", "morning", "afternoon", "evening"]]
    if display_notebook:
        from IPython.core.display_functions import display
        display(tmp)
    latex_str = (
            tmp
            .style.to_latex()
            .replace("\$", "$")
            .replace("{*}", "{15mm}")
            .replace("age_group", "Age, in years")
            .replace("biological_sex", "Reported sex")
            .replace("day_of_week", "Day of week")
            .replace("energy_group", "Move ring")
            .replace("stand_group", "Stand ring")
            .replace("exercise_group", "Exercise ring")
            .replace("\\multirow", "\\midrule\n\\multirow")
            .replace("9##", "")
    )
    return latex_str


if __name__ == '__main__':
    results = analyze_each_control_group()
    print(analysis_to_latex_table(results, effect="relative"))
    print(analysis_to_latex_table(results, effect="absolute"))
