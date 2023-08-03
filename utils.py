"""
Achille Nazaret

Contains utility functions to load and process the data.
"""
import numpy as np
import pandas as pd


def load_data(folder_path: str, control_variable: str = "All") -> pd.DataFrame:
    """
    Load the data that is in the given folder path.
    The exact file to load is determined by the `control_variable` - it is hardcoded.
    Preprocess the data to assign the integer values to the corresponding categories.
    Compute the un-normalized standing rate: `u_probability`.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing the data.
    control_variable : str
        The control variable to use. Can be one of the following:
        - "All": no control variable
        - "age_group": age group
        - "biological_sex": the biological sex
        - "day_of_week": the day of the week
        - "energy_group": the energy group
        - "exercise_group": the exercise group
        - "stand_group": the stand group

    Returns
    -------
    data : pd.DataFrame
        The loaded data.
    """

    if control_variable == "All":
        file_path = folder_path + "/data-.npz"
    elif control_variable == "age_group":
        file_path = folder_path + "/data-age-day_of_week.npz"
    elif control_variable == "biological_sex":
        file_path = folder_path + "/data-biological_sex-fixed.npz"
    elif control_variable == "day_of_week":
        file_path = folder_path + "/data-age-day_of_week.npz"
    elif control_variable == "energy_group":
        file_path = folder_path + "/data-energy_group.npz"
    elif control_variable == "exercise_group":
        file_path = folder_path + "/data-exercise_group.npz"
    elif control_variable == "stand_group":
        file_path = folder_path + "/data-stand_group.npz"
    else:
        raise ValueError("Invalid control variable")

    data = pd.DataFrame({k: v for k, v in np.load(file_path).items()})
    data["target_bucket"] = pd.to_datetime(
        data["target_bucket"].apply(lambda x: "%02d:%02d" % (x // 60, x % 60))
    )
    if "day_of_week" in data:
        data["day_of_week"] = data["day_of_week"].replace(
            {
                1: "Sunday",
                2: "Monday",
                3: "Tuesday",
                4: "Wednesday",
                5: "Thursday",
                6: "Friday",
                7: "Saturday",
            }
        )
    if "biological_sex" in data:
        data["biological_sex"] = data["biological_sex"].replace(
            {
                0: "Male",
                1: "Female",
                2: "Other",
                3: "NotSet",
            }
        )
    data["u_probability"] = (data["stand_minutes"] > 0).astype(int) * data["count"]
    return data


biological_sex_counts = {
    "biological_sex": ["Female", "Male", "Other"],
    "user-day": [16480598, 36159359, 638305 + 132576],
    "user": [57731, 105705, 2094 + 504],
}
biological_sex_counts = pd.DataFrame(biological_sex_counts).set_index("biological_sex")

day_of_week_counts = {
    "day_of_week": [
        "Monday",
        "Saturday",
        "Wednesday",
        "Friday",
        "Thursday",
        "Sunday",
        "Tuesday",
    ],
    "user-day": [
        7544532,
        7768065,
        7824785,
        7762050,
        7758296,
        7644732,
        7776724,
    ],
}
day_of_week_counts = pd.DataFrame(day_of_week_counts).set_index("day_of_week")
day_of_week_counts.loc["Weekday"] = day_of_week_counts.loc[
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
    ]
].sum()
day_of_week_counts.loc["Weekend"] = day_of_week_counts.loc[["Saturday", "Sunday"]].sum()
day_of_week_counts[
    "user"
] = "--"  # it does not make sense to compute the number of users for day of week
age_groups_counts = {
    "age_group": [
        "18 -- 24",
        "25 -- 34",
        "35 -- 44",
        "45 -- 54",
        "55 -- 64",
        "65 -- 74",
        "75 -- 84",
    ],
    "user-day": [
        4036353,
        11974775,
        16942393,
        9760343,
        6506998,
        3048398,
        1141578,
    ],
    "user": [17771, 41307, 51454, 27443, 17378, 7851, 2830],
}
age_groups_counts = pd.DataFrame(age_groups_counts).set_index("age_group")


exercise_group_counts = {
    "exercise_group": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "user-day": [
        11525507,
        7744891,
        6471099,
        5723588,
        4948000,
        4332363,
        3661338,
        2985666,
        2393536,
        2871385,
        727971,
    ],
    "user": [45429, 23878, 19320, 16485, 14090, 12294, 10194, 8028, 6365, 6886, 2583],
}

energy_group_counts = {
    "energy_group": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "user-day": [
        4848598,
        3957024,
        4580394,
        5083102,
        5767536,
        6607638,
        7227726,
        5322597,
        3975212,
        4830203,
        1185314,
    ],
    "user": [22231, 14214, 15349, 16032, 17272, 19107, 19491, 14723, 11034, 11828, 4271],
}

stand_group_counts = {
    "stand_group": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "user-day": [
        777803,
        1485467,
        2310331,
        3225974,
        4038883,
        5085261,
        6103811,
        7126351,
        8604743,
        13289337,
        1337383,
    ],
    "user": [6481, 7585, 9962, 12107, 13828, 16424, 18285, 20140, 22961, 32469, 5310],
}
stand_group_counts = pd.DataFrame(stand_group_counts).set_index("stand_group")
exercise_group_counts = pd.DataFrame(exercise_group_counts).set_index("exercise_group")
energy_group_counts = pd.DataFrame(energy_group_counts).set_index("energy_group")


for df_tmp in [stand_group_counts, exercise_group_counts, energy_group_counts]:
    df_tmp.loc["0 -- 49"] = df_tmp.loc[[0, 1, 2, 3, 4]].sum()
    df_tmp.loc["50 -- 79"] = df_tmp.loc[[5, 6, 7]].sum()
    df_tmp.loc["80 -- 99"] = df_tmp.loc[[8, 9]].sum()
    df_tmp.loc["100"] = df_tmp.loc[[10]].sum()


def get_n_users_m_days(subgroup: str, key: str) -> str:
    """Returns the number of users in a subgroup.

    Parameters
    ----------
    subgroup : str
        The subgroup.
    key : str
        Can be:
        - "user": the number of distinct users in the specific subgroup is returned.
        - "user-day": the number of distinct users-days pairs in the specific subgroup is returned.

    Returns
    -------
    str
        The number of users or user-days in the subgroup, as a string.
    """
    res = -1
    if subgroup == "--":
        # this is the total number of users or user-days, no specific subgroup
        res = age_groups_counts[key].sum()
    # otherwise, we look for the subgroup in the different dataframes
    for df in [
        age_groups_counts,
        biological_sex_counts,
        day_of_week_counts,
        exercise_group_counts,
        stand_group_counts,
        energy_group_counts,
    ]:
        if subgroup in df.index:
            res = df.loc[subgroup][key]
            break

    if res != -1:
        if type(res) != str:
            res = f"{res:,}"
        return res
    else:
        raise ValueError(f"Unknown subgroup {subgroup}")
