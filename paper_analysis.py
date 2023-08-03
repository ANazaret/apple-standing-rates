import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from utils import load_data, get_n_users_m_days
from regression_discontinuity import fit_regression_discontinuity

DATA_PATH = "../data"

CONTROL_SUBGROUPS_TABLE = [
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
    #  group 9##100 is group 100, but written like this for alphabetical order sorting ...
    #  9## is removed in the table
    (
        "energy_group",
        {"0 -- 49": [0, 1, 2, 3, 4], "50 -- 79": [5, 6, 7], "80 -- 99": [8, 9], "9##100": [10]},
    ),
    (
        "exercise_group",
        {"0 -- 49": [0, 1, 2, 3, 4], "50 -- 79": [5, 6, 7], "80 -- 99": [8, 9], "9##100": [10]},
    ),
    (
        "stand_group",
        {"0 -- 49": [0, 1, 2, 3, 4], "50 -- 79": [5, 6, 7], "80 -- 99": [8, 9], "9##100": [10]},
    ),
]

CONTROL_SUBGROUPS_FIGURE = CONTROL_SUBGROUPS_TABLE.copy()
CONTROL_SUBGROUPS_FIGURE[3] = (
    "day_of_week",
    {
        "Monday": ["Monday"],
        "Tuesday": ["Tuesday"],
        "Wednesday": ["Wednesday"],
        "Thursday": ["Thursday"],
        "Friday": ["Friday"],
        "Saturday": ["Saturday"],
        "Sunday": ["Sunday"],
    },
)


def analysis_period_of_day(model: str = "linear") -> pd.DataFrame:
    """
    Analyze standing rate in morning/afternoon/evening for
    each control group to (re)produce the tables of the paper.

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

    for control_variable, subgroups in CONTROL_SUBGROUPS_TABLE:
        df = load_data(DATA_PATH, control_variable=control_variable, delta_min=7, delta_max=15)
        for k, v in subgroups.items():
            if control_variable is not None and control_variable != "All":
                data_tmp = df[df[control_variable].isin(v)].copy()
            else:
                data_tmp = df

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
        lambda x: f'\${np.round(x["causal_effect"]*100,2)} ~ \scriptstyle \\pm '
        + str(
            max(
                np.round((x["causal_confint_high"] - x["causal_effect"]) * 100, 2),
                np.round((x["causal_effect"] - x["causal_confint_low"]) * 100, 2),
            )
        )
        + "\$",
        axis=1,
    )
    results.index.names = ["variable", "subgroup", "time_of_day"]

    return results


def analysis_to_latex_table(
    results: pd.DataFrame, effect: str = "relative", display_notebook=False
) -> str:
    """Converts the results of the analysis to a LaTeX table.

    Parameters
    ----------
    results : pd.DataFrame
        The results of the analysis.
    effect : str, optional
        The effect to report. Can be either "relative" or "absolute", by default "relative".
    display_notebook : bool, optional
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
    tmp["$n$ users"] = tmp.reset_index()["subgroup"].apply(get_n_users_m_days, key="user").values
    tmp["$m$ user-days"] = (
        tmp.reset_index()["subgroup"].apply(get_n_users_m_days, key="user-day").values
    )
    tmp = tmp[["$n$ users", "$m$ user-days", "morning", "afternoon", "evening"]]
    if display_notebook:
        from IPython.core.display_functions import display

        display(tmp)
    latex_str = (
        tmp.style.to_latex()
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


def analysis_throughout_day(
    control_variables, target_minutes=[50], hour_range=(6, 24), model="linear"
):
    """
    Analyze standing rate each hour of the day for
    each control group to (re)produce the figures of the paper.

    Parameters
    ----------
    model : str, optional (default="linear")
        The model to use for the regression discontinuity. Either "linear" or "logistic".

    Returns
    -------
    results : pd.DataFrame
        The results of the analysis.
    """

    results_throughout_day = dict()

    for control_variable, subgroups in CONTROL_SUBGROUPS_FIGURE:
        if control_variable not in control_variables:
            continue
        df = load_data(DATA_PATH, control_variable, delta_min=7, delta_max=15)
        for subgroup_name, v in subgroups.items():
            if control_variable is not None and control_variable != "All":
                if type(v) != list:
                    # TODO
                    print(", not remove list v")
                    v = [v]
                data_tmp = df[df[control_variable].isin(v)].copy()
            else:
                data_tmp = df
            data_tmp = (
                data_tmp.groupby(["delta", "target_bucket", "stand"])["count"].sum().reset_index()
            )

            for m in target_minutes:
                for h in range(*hour_range):
                    if m == "other":
                        target_minutes_tmp = [5, 10, 15, 20, 25, 30, 35, 40, 45]
                        nudge_delta = 11
                        m = -1
                    else:
                        target_minutes_tmp = [m]
                        if m == 55:
                            nudge_delta = 12
                        else:
                            nudge_delta = 11
                    key = (control_variable, subgroup_name, h, m)

                    results_throughout_day[key] = fit_regression_discontinuity(
                        data_tmp,
                        target_hours=[h],
                        target_minutes=target_minutes_tmp,
                        nudge_at_delta=nudge_delta,
                        model=model,
                    )

    results_throughout_day = pd.DataFrame(results_throughout_day).T

    results_throughout_day.index.names = ["control_variable", "subgroup_name", "hour", "minute"]
    results_throughout_day["time"] = pd.to_datetime(
        results_throughout_day.reset_index().hour.astype(str)
        + ":"
        + results_throughout_day.reset_index().minute.astype(str)
    ).values

    return results_throughout_day


def make_figure_5():
    target_minutes = [50, 55]
    results_throughout_day = analysis_throughout_day(
        control_variables=["All"], target_minutes=target_minutes
    )

    fig, axs = plt.subplots(1, 2, figsize=[10, 2.7], dpi=180)

    def get_interval_label(m):
        if m != -1:
            return "[%d-%d]" % (m, m + 4)
        else:
            return "[0-49]"

    data_plot = results_throughout_day.reset_index()
    ax = axs[0]

    line_plot = sns.lineplot(
        data=data_plot,
        x="time",
        y="causal_effect",
        hue="minute",
        style="minute",
        ax=ax,
        hue_order=target_minutes,
        style_order=target_minutes,
        palette=sns.color_palette(n_colors=len(target_minutes)),
    )

    for m in target_minutes:
        ax.fill_between(
            data_plot[data_plot["minute"] == m]["time"],
            data_plot[data_plot["minute"] == m]["causal_confint_low"],
            data_plot[data_plot["minute"] == m]["causal_confint_high"],
            alpha=0.2,
        )
        ax.plot(
            data_plot[data_plot["minute"] == m]["time"],
            data_plot[data_plot["minute"] == m]["causal_confint_low"],
            c="black",
            lw=0.30,
        )
        ax.plot(
            data_plot[data_plot["minute"] == m]["time"],
            data_plot[data_plot["minute"] == m]["causal_confint_high"],
            c="black",
            lw=0.30,
        )
    ax.grid(axis="y")
    ax.grid(axis="x", linestyle=":")
    ax.set_ylabel("Absolute causal effect of the\n nudge on the standing rate")
    ax.set_xlabel("Time of day")
    leg = ax.legend(
        title="Interval past\nthe hour", labels=[get_interval_label(m) for m in target_minutes]
    )
    plt.setp(leg.get_title(), multialignment="center")

    formatter = matplotlib.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(formatter)
    locator = matplotlib.dates.HourLocator(interval=4)
    ax.xaxis.set_major_locator(locator)
    locator = matplotlib.dates.HourLocator(interval=1)
    ax.xaxis.set_minor_locator(locator)

    ax = axs[1]

    line_plot = sns.lineplot(
        data=data_plot,
        x="time",
        y="relative",
        hue="minute",
        style="minute",
        ax=ax,
        hue_order=target_minutes,
        style_order=target_minutes,
        palette=sns.color_palette(n_colors=len(target_minutes)),
    )
    for m in target_minutes:
        ax.fill_between(
            data_plot[data_plot["minute"] == m]["time"],
            data_plot[data_plot["minute"] == m]["relative_low"],
            data_plot[data_plot["minute"] == m]["relative_high"],
            alpha=0.2,
        )
        ax.plot(
            data_plot[data_plot["minute"] == m]["time"],
            data_plot[data_plot["minute"] == m]["relative_low"],
            c="black",
            lw=0.30,
        )
        ax.plot(
            data_plot[data_plot["minute"] == m]["time"],
            data_plot[data_plot["minute"] == m]["relative_high"],
            c="black",
            lw=0.30,
        )
    ticks = np.arange(-0.2, 0.8, 0.1)
    ax.set_yticks(
        ticks,
        [
            "+%2.0f%%" % (t * 100) if t > 0 else "0" if t == 0 else "%2.0f%%" % (t * 100)
            for t in ticks
        ],
    )
    # ticks = np.arange(-0.25, 0.8, 0.1)
    # ax.set_yticks(
    #     ticks,
    #     minor=True
    # )
    ax.set_ylim(-0.05, 0.6)
    # ax.set_xlim(pd.Timestamp("06:00:00"), pd.Timestamp("23:59:59"))

    ax.grid(axis="x", linestyle=":")
    ax.grid(axis="y")
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Relative causal effect of the\n nudge on the standing rate")
    leg = ax.legend(
        title="Interval past\nthe hour", labels=[get_interval_label(m) for m in target_minutes]
    )
    plt.setp(leg.get_title(), multialignment="center")

    formatter = matplotlib.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(formatter)
    locator = matplotlib.dates.HourLocator(interval=4)
    ax.xaxis.set_major_locator(locator)
    locator = matplotlib.dates.HourLocator(interval=1)
    ax.xaxis.set_minor_locator(locator)
    fig.tight_layout(pad=0.1, w_pad=1.2)


def make_figure_6():

    target_minutes = [50]
    fig, axs = plt.subplots(3, 2, figsize=[8, 8], dpi=180)
    plot_layout = [
        ("age_group", axs[0, 0]),
        ("biological_sex", axs[0, 1]),
        ("day_of_week", axs[1, 0]),
        ("energy_group", axs[1, 1]),
        ("exercise_group", axs[2, 0]),
        ("stand_group", axs[2, 1]),
    ]

    results_throughout_day = analysis_throughout_day(
        control_variables=[v[0] for v in plot_layout],
        target_minutes=target_minutes,
        hour_range=[7, 24],
    )

    for control_variable, ax in plot_layout:
        control_variables_palette = {
            "age_group": sns.color_palette("viridis_r", as_cmap=True)(
                np.linspace(0, 1, len(dict(CONTROL_SUBGROUPS_FIGURE)["age_group"]))
            ),
            "biological_sex": None,
            "day_of_week": sns.color_palette("cool", n_colors=7),
            "exercise_group": sns.color_palette("plasma_r", as_cmap=True)(
                np.linspace(0, 1, len(dict(CONTROL_SUBGROUPS_FIGURE)["exercise_group"]))
            ),
            "energy_group": sns.color_palette("plasma_r", as_cmap=True)(
                np.linspace(0, 1, len(dict(CONTROL_SUBGROUPS_FIGURE)["energy_group"]))
            ),
            "stand_group": sns.color_palette("plasma_r", as_cmap=True)(
                np.linspace(0, 1, len(dict(CONTROL_SUBGROUPS_FIGURE)["stand_group"]))
            ),
        }
        palette = control_variables_palette[control_variable]

        data = results_throughout_day.reset_index()
        data = data[data["control_variable"] == control_variable]

        line_plot = sns.lineplot(
            data=data,
            x="time",
            y="relative",
            hue="subgroup_name",
            ax=ax,
            palette=palette,
        )
        for i, v in enumerate(data["subgroup_name"].unique()):
            ax.fill_between(
                data[data["subgroup_name"] == v]["time"],
                data[data["subgroup_name"] == v]["relative_low"],
                data[data["subgroup_name"] == v]["relative_high"],
                color=line_plot.get_lines()[i].get_color(),
                alpha=0.2,
            )
        ax.legend(ncol=3, title=control_variable)

        def hide_evey_other(l):
            return [ll if i % 2 == 1 else "" for i, ll in enumerate(l)]

        control_variables_ticks = {
            "age_group": np.arange(-0.25, 2, 0.25),
            "biological_sex": np.arange(-0.25, 2, 0.25),
            "day_of_week": np.arange(-0.25, 2, 0.25),
            "exercise_group": np.arange(-0.5, 4.6, 0.5),
            "stand_group": np.arange(-0.5, 4.6, 0.5),
            "energy_group": np.arange(-0.5, 4.6, 0.5),
        }
        control_variables_ymax = {
            "age_group": 1.75,
            "biological_sex": 1.75,
            "day_of_week": 1.75,
            "exercise_group": 3.5,
            "stand_group": 3.5,
            "energy_group": 3.5,
        }
        control_variables_ymin = {
            "age_group": -0.25,
            "biological_sex": -0.25,
            "day_of_week": -0.25,
            "exercise_group": -0.5,
            "stand_group": -0.5,
            "energy_group": -0.5,
        }

        ax.set_yticks(
            control_variables_ticks[control_variable],
            hide_evey_other(
                [
                    "+%2.0f%%" % (t * 100) if t > 0 else "0%" if t == 0 else "%2.0f%%" % (t * 100)
                    for t in control_variables_ticks[control_variable]
                ]
            ),
        )
        ax.set_ylim(
            bottom=control_variables_ymin[control_variable],
            top=control_variables_ymax[control_variable],
        )

        ax.grid(axis="x", linestyle=":")
        ax.grid(axis="y")
        ax.set_xlabel("Time of day")

        legend_title = {
            "age_group": "Age",
            "biological_sex": "Reported Sex",
            "day_of_week": "Day of Week",
            "exercise_group": "Exercise Ring",
            "stand_group": "Stand Ring",
            "energy_group": "Move Ring",
        }
        legend_ncol = {
            "age_group": 4,
            "biological_sex": 1,
            "exercise_group": 1,
            "stand_group": 1,
            "energy_group": 1,
            "day_of_week": 3,
        }
        legend_loc = {
            "age_group": "upper center",
            "biological_sex": "upper right",
            "exercise_group": "upper right",
            "stand_group": "upper right",
            "energy_group": "upper right",
            "day_of_week": "upper center",
        }

        ax.set_ylabel("")

        legend_names = {
            "age_group": ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84"],
            "biological_sex": ["Male", "Female", "Other"],
            "day_of_week": [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            "exercise_group": [
                "0-49%",
                "50-79%",
                "80-99%",
                "100%",
            ],
            "stand_group": [
                "0-49%",
                "50-79%",
                "80-99%",
                "100%",
            ],
            "energy_group": [
                "0-49%",
                "50-79%",
                "80-99%",
                "100%",
            ],
        }

        legend_n_cols = legend_ncol[control_variable]

        # reorder_legend
        handles = ax.get_legend().legendHandles
        indices = []
        for i in range(legend_n_cols):
            indices.extend(range(i, len(legend_names[control_variable]), legend_n_cols))

        handles = [handles[i] for i in indices]
        labels = [legend_names[control_variable][i] for i in indices]
        ax.legend(
            handles,
            labels,
            ncol=legend_n_cols,
            title=legend_title[control_variable],
            fontsize=8.5,
            loc=legend_loc[control_variable],
            title_fontsize=9,
            labelspacing=0.3,
            borderpad=0.6,
            columnspacing=1,
            handletextpad=0.5,
        )

        formatter = matplotlib.dates.DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(formatter)
        locator = matplotlib.dates.HourLocator(interval=4)
        ax.xaxis.set_major_locator(locator)
        locator = matplotlib.dates.HourLocator(interval=1)
        ax.xaxis.set_minor_locator(locator)

    fig.tight_layout()


if __name__ == "__main__":
    results_analysis = analysis_period_of_day()
    print(analysis_to_latex_table(results_analysis, effect="relative"))
    print(analysis_to_latex_table(results_analysis, effect="absolute"))
    make_figure_5()
    # plt.savefig(f"figure5.pdf", bbox_inches="tight")
    plt.show()
    make_figure_6()
    # plt.savefig(f"figure6.pdf", bbox_inches="tight")
    plt.show()
