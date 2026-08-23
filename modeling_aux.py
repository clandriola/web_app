# has feature engineering and rule-based modeling


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Polygon


# Behavioral Features Functions

LANE_LINES = np.array([-60, -48, -36, -24, -12])
LANE_WIDTH = 12.0
G = 32.174

def B0_unknowns(df):
    """
    Returns True if at least one vehicle is touching a lane line for all
    timestamps in the trajectory.

    A vehicle is considered touching a lane line when any lane line falls
    within its lateral footprint:
        [y - width/2, y + width/2]
    """
    for veh in ["i", "j"]:
        touching_all = True

        for _, row in df.iterrows():
            y = row[f"y_{veh}"]
            w = row[f"width_{veh}"] / 2.0

            touching = np.any((LANE_LINES >= y - w) & (LANE_LINES <= y + w))

            if not touching:
                touching_all = False
                break

        if touching_all:
            return True

    return False


def B1_same_lane(df):
    """
    Returns True if v1 and v2 overlap laterally at both t=1 and t=2 s.
    """

    idx0 = (df["t"] - 0.0).abs().idxmin()
    idx2 = (df["t"] - 2.0).abs().idxmin()

    for row in [df.loc[idx0], df.loc[idx2]]:

        i_min = row["y_i"] - row["width_i"] / 2.0
        i_max = row["y_i"] + row["width_i"] / 2.0

        j_min = row["y_j"] - row["width_j"] / 2.0
        j_max = row["y_j"] + row["width_j"] / 2.0

        overlap = max(i_min, j_min) <= min(i_max, j_max)

        if not overlap:
            return False

    return True


def B2_v1_decelerating(df):
    """
    Returns True if v1 is decelerating during the first 3 seconds.

    Conditions:
    - acc_i < 0 for all timestamps in first 3 seconds.
    - acc_i < -0.125*G for at least one timestamp.
    """
    df3 = df[df["t"] <= 3.0]

    return (
        (df3["acc_i"] < 0).all()
        and (df3["acc_i"] < -0.125 * G).any()
    )


def B3_crossing_paths(df):
    """
    Returns True if the vehicles cross paths.

    Conditions:
    - The vehicle that starts on the right ends on the left (and vice versa).
    - The lateral gap between the vehicles at t=0 and t=6 is greater than -2 ft.
      Negative gaps indicate overlap; values above -2 ft allow only small overlap.
    """
    first = df.iloc[0]
    last = df.iloc[-1]

    # Right/left vehicle at t=0
    if abs(first["y_i"]) > abs(first["y_j"]):
        right_0 = "i"
        left_0 = "j"
    else:
        right_0 = "j"
        left_0 = "i"

    # Right/left vehicle at t=6
    if abs(last["y_i"]) > abs(last["y_j"]):
        right_6 = "i"
        left_6 = "j"
    else:
        right_6 = "j"
        left_6 = "i"

    # Gap at t=0
    i_min = first["y_i"] - first["width_i"] / 2.0
    i_max = first["y_i"] + first["width_i"] / 2.0
    j_min = first["y_j"] - first["width_j"] / 2.0
    j_max = first["y_j"] + first["width_j"] / 2.0

    gap_0 = max(i_min, j_min) - min(i_max, j_max)

    # Gap at t=6
    i_min = last["y_i"] - last["width_i"] / 2.0
    i_max = last["y_i"] + last["width_i"] / 2.0
    j_min = last["y_j"] - last["width_j"] / 2.0
    j_max = last["y_j"] + last["width_j"] / 2.0

    gap_6 = max(i_min, j_min) - min(i_max, j_max)

    vehicles_switched = (right_0 == left_6) and (left_0 == right_6)

    return vehicles_switched and gap_0 > -2.0 and gap_6 > -2.0


def B4_y_proximity(df):
    """
    Returns True if the lateral gap between vehicles at t=0 is smaller
    than one lane width.

    Vehicle footprints are approximated using:
        y +/- width/2
    """
    row = df.iloc[0]

    i_min = row["y_i"] - row["width_i"] / 2.0
    i_max = row["y_i"] + row["width_i"] / 2.0

    j_min = row["y_j"] - row["width_j"] / 2.0
    j_max = row["y_j"] + row["width_j"] / 2.0

    if i_max < j_min:
        gap = j_min - i_max
    elif j_max < i_min:
        gap = i_min - j_max
    else:
        gap = 0.0

    return gap < LANE_WIDTH


def B5_v1_in_front(df):
    """
    Returns True if v1 is in front at t=0.

    Returns False if vehicle i is fully inside the same lane
    at both t=0 and t=6.
    """
    first = df.iloc[0]
    last = df.iloc[-1]
    if B3_crossing_paths(df):
        if not B7_v_changing_lanes(df, "j"):
            return True
        else: 
            touching_t0 = np.any(
                (LANE_LINES >= first["y_i"] - first["width_i"] / 2.0) &
                (LANE_LINES <= first["y_i"] + first["width_i"] / 2.0)
            )

            touching_t6 = np.any(
                (LANE_LINES >= last["y_i"] - last["width_i"] / 2.0) &
                (LANE_LINES <= last["y_i"] + last["width_i"] / 2.0)
            )

            if not touching_t0 and not touching_t6:

                lane_t0 = np.sum(first["y_i"] < LANE_LINES)
                lane_t6 = np.sum(last["y_i"] < LANE_LINES)

                if lane_t0 == lane_t6:
                    return False

    return first["x_i"] > first["x_j"]
    

def B6_v_from_right(df, vehicle):
    """
    Returns True if the specified vehicle ('i' or 'j') starts on the right.

    Since y goes from 0 (left) to -60 (right), the vehicle on the right
    has the largest absolute y value.
    """
    row = df.iloc[0]

    if vehicle == "i":
        return abs(row["y_i"]) > abs(row["y_j"])
    else:
        return abs(row["y_j"]) > abs(row["y_i"])


def B7_v_changing_lanes(df, vehicle):
    """
    Returns True if the specified vehicle ('i' or 'j') is changing lanes
    between t=0 and t=4.

    Conditions:
    - Vehicle touches a lane line.
    - Heading points toward the other vehicle.
      * other vehicle on right  -> hy < 0
      * other vehicle on left   -> hy > 0
    """
    other = "j" if vehicle == "i" else "i"

    df4 = df[df["t"] <= 4.0]

    for _, row in df4.iterrows():

        y = row[f"y_{vehicle}"]
        w = row[f"width_{vehicle}"] / 2.0

        touching = np.any(
            (LANE_LINES >= y - w) &
            (LANE_LINES <= y + w)
        )

        if not touching:
            continue

        other_on_right = abs(row[f"y_{other}"]) > abs(row[f"y_{vehicle}"])

        if other_on_right and row[f"hy_{vehicle}"] < 0:
            return True

        if not other_on_right and row[f"hy_{vehicle}"] > 0:
            return True

    return False


def B8_v_passing(df, vehicle):
    """
    Returns True if the specified vehicle ('i' or 'j') passes the other.

    A pass occurs when:
    - vehicle starts behind the other in x at t=0
    - vehicle ends ahead of the other at t=6
    """
    other = "j" if vehicle == "i" else "i"

    first = df.iloc[0]
    last = df.iloc[-1]

    return (
        first[f"x_{vehicle}"] < first[f"x_{other}"]
        and
        last[f"x_{vehicle}"] > last[f"x_{other}"]
    )


def B9_test_steer(df, vehicle):
    """
    Returns:
    - "left"     : lane change away from other vehicle toward the left
    - "right"    : lane change away from other vehicle toward the right
    - "centered" : stayed within 3 ft of a lane center for t=0..6
    - "NA"       : otherwise
    """
    other = "j" if vehicle == "i" else "i"

    # Existing left/right logic
    df4 = df[df["t"] >= 3.0]

    for _, row in df4.iterrows():

        y = row[f"y_{vehicle}"]
        w = row[f"width_{vehicle}"] / 2.0

        touching = np.any(
            (LANE_LINES >= y - w) &
            (LANE_LINES <= y + w)
        )

        if not touching:
            continue

        other_on_right = abs(row[f"y_{other}"]) > abs(row[f"y_{vehicle}"])
        hy = row[f"hy_{vehicle}"]

        # Going away from vehicle on the right -> steer left
        if other_on_right and hy > 0:
            return "left"

        # Going away from vehicle on the left -> steer right
        if not other_on_right and hy < 0:
            return "right"

    # ------------------------------------------------------------------
    # If neither left nor right, check whether vehicle stayed within
    # 3 ft of the center of any lane for the entire time 0 <= t <= 6
    # ------------------------------------------------------------------
    df06 = df[(df["t"] >= 0.0) & (df["t"] <= 6.0)]

    lane_centers = (LANE_LINES[:-1] + LANE_LINES[1:]) / 2.0

    y_vals = df06[f"y_{vehicle}"].to_numpy()

    for center in lane_centers:
        if np.all(np.abs(y_vals - center) <= 1.5):
            return "no"

    return "NA"


def B10_test_acceleration_simple(df, vehicle):
    """
    Evaluate acceleration/braking between t=2 and t=4.

    Returns:
    - "acc" : jerk > 0.1 g/s AND acceleration >= 1.25 g
    - "dec" : jerk < -0.1 g/s AND acceleration <= -1.25 g
    - "NA"  : otherwise
    """

    df2 = df[(df["t"] >= 1.5) & (df["t"] <= 4)]

    if len(df2) < 2:
        return "NA"

    acc = df2[f"acc_{vehicle}"]

    # jerk in g/s
    jerk = acc.diff() / (G * 0.04)

    has_acc = (
        (jerk > 0.1).any()
        and (acc >= 0.125 * G).any()
    )

    has_dec = (
        (jerk < -0.1).any()
        and (acc <= -0.125 * G).any()
    )

    if has_acc and not has_dec:
        return "acc"

    if has_dec and not has_acc:
        return "dec"

    return "NA"


def B11_same_lane(df, vehicle, t=6.0):
    """
    Returns True if the vehicle is fully within a lane at time t.
    Returns False if any part of the vehicle touches a lane line.
    """

    idx = (df["t"] - t).abs().idxmin()
    row = df.loc[idx]

    y = row[f"y_{vehicle}"]
    w = row[f"width_{vehicle}"] / 2.0

    touching = np.any(
        (LANE_LINES >= y - w) &
        (LANE_LINES <= y + w)
    )

    return not touching


def B12_v_changing_lanes_conflict(df, vehicle):
    """
    Returns True if the specified vehicle ('i' or 'j') is changing lanes
    between t=0 and t=4.

    Conditions:
    - Vehicle touches a lane line.
    - Heading points toward the other vehicle.
      * other vehicle on right  -> hy < 0
      * other vehicle on left   -> hy > 0
    """
    other = "j" if vehicle == "i" else "i"

    df4 = df[(df["t"] >= 2) & (df["t"] <= 4)]

    for _, row in df4.iterrows():

        y = row[f"y_{vehicle}"]
        w = row[f"width_{vehicle}"] / 2.0

        touching = np.any(
            (LANE_LINES >= y - w) &
            (LANE_LINES <= y + w)
        )

        if not touching:
            continue

        other_on_right = abs(row[f"y_{other}"]) > abs(row[f"y_{vehicle}"])

        if other_on_right and row[f"hy_{vehicle}"] < 0:
            return True

        if not other_on_right and row[f"hy_{vehicle}"] > 0:
            return True

    return False


def B13_y_and_heading_test(df):
    """
    Returns True if BOTH conditions are satisfied:

    1. Vehicle i's lateral position varies less than 3 ft
       between t=2 and t=3.

    2. The absolute heading of vehicle i is smaller than
       the absolute heading of vehicle j at t=3.
    """

    # Closest samples to t=2 and t=3
    idx2 = (df["t"] - 0.0).abs().idxmin()
    idx3 = (df["t"] - 3.0).abs().idxmin()

    row2 = df.loc[idx2]
    row3 = df.loc[idx3]

    # Condition 1: y variation < 3 ft
    y_change_ok = abs(row3["y_i"] - row2["y_i"]) < 3.0

    # Condition 2: |heading_i| < |heading_j| at t=3
    heading_ok = abs(row3["hy_i"]) < abs(row3["hy_j"])

    return y_change_ok and heading_ok



# Physical + Behavioral Features

def feature_eng_timestamp(df):
    # --------------------------
    # Relative position
    # --------------------------
    df["dx"] = df["x_j"] - df["x_i"]
    df["dy"] = df["y_j"] - df["y_i"]

    # --------------------------
    # Euclidean distance
    # --------------------------
    df["d"] = np.sqrt(df["dx"]**2 + df["dy"]**2)

    # --------------------------
    # Speed of each vehicle
    # --------------------------
    df["v_i"] = np.sqrt(df["vx_i"]**2 + df["vy_i"]**2)
    df["v_j"] = np.sqrt(df["vx_j"]**2 + df["vy_j"]**2)

    # --------------------------
    # Relative velocity
    # --------------------------
    df["dvx"] = df["vx_j"] - df["vx_i"]
    df["dvy"] = df["vy_j"] - df["vy_i"]

    df["dv"] = np.sqrt(df["dvx"]**2 + df["dvy"]**2)

    # --------------------------
    # Closing speed
    # Positive = approaching
    # Negative = separating
    # --------------------------
    df["cv"] = -(
        (df["dvx"] * df["dx"] + df["dvy"] * df["dy"])
        / df["d"]
    )

    # --------------------------
    # Relative acceleration
    # --------------------------
    df["da"] = df["acc_j"] - df["acc_i"]

    # --------------------------
    # Gap
    # Center distance minus half lengths
    # --------------------------
    df["gap"] = (
        df["d"]
        - (df["length_i"] / 2)
        - (df["length_j"] / 2)
    )

    # --------------------------
    # Jerk
    # dt = time step
    # --------------------------
    dt = df["t"].diff()
    
    df["j_i"] = (df["acc_i"].diff() / dt).fillna(0)
    df["j_j"] = (df["acc_j"].diff() / dt).fillna(0)

    return df


def feature_eng_pair(df):

    dt = np.median(np.diff(df["t"]))

    features = {}

    # ==================================================
    # CRITICAL WINDOWS
    # ==================================================
    windows = {
        "02": (0, 2),
        "24": (2, 4),
        "46": (4, 6),
        "23": (2, 3),
        "35": (3, 5),
    }

    for suffix, (t0, t1) in windows.items():

        df_w = df[(df["t"] >= t0) & (df["t"] < t1)].copy()

        # ----------------------------------------------
        # VEHICLE i
        # ----------------------------------------------
        features[f"j_i_max_{suffix}"] = df_w["j_i"].max()
        features[f"j_i_min_{suffix}"] = df_w["j_i"].min()

        features[f"acc_i_max_{suffix}"] = df_w["acc_i"].max()
        features[f"acc_i_min_{suffix}"] = df_w["acc_i"].min()

        features[f"jerk_i_pos_time_{suffix}"] = (
            (df_w["j_i"] > 0).sum() * dt
        )

        features[f"jerk_i_neg_time_{suffix}"] = (
            (df_w["j_i"] < 0).sum() * dt
        )

        features[f"acc_i_pos_time_{suffix}"] = (
            (df_w["acc_i"] > 0).sum() * dt
        )

        features[f"acc_i_neg_time_{suffix}"] = (
            (df_w["acc_i"] < 0).sum() * dt
        )

        features[f"acc_i_mean_{suffix}"] = (
            df_w["acc_i"].mean()
        )

        features[f"acc_i_range_abs_{suffix}"] = abs(
            df_w["acc_i"].max() - df_w["acc_i"].min()
        )

        # ----------------------------------------------
        # VEHICLE j
        # ----------------------------------------------
        features[f"j_j_max_{suffix}"] = df_w["j_j"].max()
        features[f"j_j_min_{suffix}"] = df_w["j_j"].min()

        features[f"acc_j_max_{suffix}"] = df_w["acc_j"].max()
        features[f"acc_j_min_{suffix}"] = df_w["acc_j"].min()

        features[f"jerk_j_pos_time_{suffix}"] = (
            (df_w["j_j"] > 0).sum() * dt
        )

        features[f"jerk_j_neg_time_{suffix}"] = (
            (df_w["j_j"] < 0).sum() * dt
        )

        features[f"acc_j_pos_time_{suffix}"] = (
            (df_w["acc_j"] > 0).sum() * dt
        )

        features[f"acc_j_neg_time_{suffix}"] = (
            (df_w["acc_j"] < 0).sum() * dt
        )

        features[f"acc_j_mean_{suffix}"] = (
            df_w["acc_j"].mean()
        )

        features[f"acc_j_range_abs_{suffix}"] = abs(
            df_w["acc_j"].max() - df_w["acc_j"].min()
        )

    # ==================================================
    # STATIC VEHICLE FEATURES
    # ==================================================
    features["length_i"] = df["length_i"].iloc[0]
    features["width_i"] = df["width_i"].iloc[0]

    features["length_j"] = df["length_j"].iloc[0]
    features["width_j"] = df["width_j"].iloc[0]

    features["y_i_max_recovery"] = max(
        max(df["y_i"].iloc[k:].max() - df["y_i"].iloc[k] for k in range(len(df))),
        max(df["y_i"].iloc[k] - df["y_i"].iloc[k:].min() for k in range(len(df)))
    )
    features["y_j_max_recovery"] = max(
        max(df["y_j"].iloc[k:].max() - df["y_j"].iloc[k] for k in range(len(df))),
        max(df["y_j"].iloc[k] - df["y_j"].iloc[k:].min() for k in range(len(df)))
    )
    up_recovery = max(
        df["y_i"].iloc[k:].max() - df["y_i"].iloc[k]
        for k in range(len(df))
    )
    down_recovery = max(
        df["y_i"].iloc[k] - df["y_i"].iloc[k:].min()
        for k in range(len(df))
    )
    features["y_i_max_recovery_signed"] = (
        up_recovery if up_recovery > down_recovery
        else -down_recovery
    )

    features["same_heading_time"] = (
    (np.sign(df["hy_i"]) == np.sign(df["hy_j"])).sum() * dt)
    

    features["t_closest"] = df.loc[df["d"].idxmin(), "t"]

    # ==================================================
    # SSM
    # ==================================================
    features["TTC_min"] = df.loc[df["TTC"] > 0, "TTC"].min()
    features["MTTC_min"] = df.loc[df["MTTC"] > 0, "MTTC"].min()
    features["DRAC_max"] = df.loc[df["DRAC"] >= 0, "DRAC"].max()

    # ==================================================
    # SNAPSHOTS AT t = 0,1,2,3,4,5,6
    # ==================================================
    target_times = [0, 1, 2, 3, 4, 5, 6]

    for tt in target_times:

        idx = (df["t"] - tt).abs().idxmin()
        row = df.loc[idx]

        features[f"dx_t{tt}"] = row["dx"]
        features[f"dy_t{tt}"] = row["dy"]
        #features[f"d_t{tt}"] = row["d"]

        #features[f"dvx_t{tt}"] = row["dvx"]
        #features[f"dvy_t{tt}"] = row["dvy"]
        features[f"dv_t{tt}"] = row["dv"]

        features[f"hyi_t{tt}"] = row["hy_i"]
        features[f"hyj_t{tt}"] = row["hy_j"]
        features[f"acci_t{tt}"] = row["acc_i"]
        features[f"accj_t{tt}"] = row["acc_j"]
        features[f"vi_t{tt}"] = row["vx_i"]
        features[f"vj_t{tt}"] = row["vx_j"]

        features[f"same_lane_i_{tt}"] = int(B11_same_lane(df, "i", tt))
        features[f"same_lane_j_{tt}"] = int(B11_same_lane(df, "j", tt))


    if features[f"same_lane_i_{4}"] and features[f"same_lane_i_{5}"] and features[f"same_lane_i_{6}"]:
        features[f"same_lane_end"] = True
    else:
        features[f"same_lane_end"] = False
    features[f"B12_i_changing_lanes_conflict"] = B12_v_changing_lanes_conflict(df, "i")
    features[f"B12_j_changing_lanes_conflict"] = B12_v_changing_lanes_conflict(df, "j")
    features[f"B13_y_and_heading_test"] = B13_y_and_heading_test(df)

    # ==================================================
    # RULE FEATURES
    # ==================================================
    rule_features = {
        "B1_same_lane": int(B1_same_lane(df)),
        "B2_v1_decelerating": int(B2_v1_decelerating(df)),
        "B3_crossing_paths": int(B3_crossing_paths(df)),
        "B4_y_proximity": int(B4_y_proximity(df)),
        "B5_v1_in_front": int(B5_v1_in_front(df)),
        "B6_i_from_right": int(B6_v_from_right(df, "i")),
        "B6_j_from_right": int(B6_v_from_right(df, "j")),
        "B7_i_changing_lanes": int(B7_v_changing_lanes(df, "i")),
        "B7_j_changing_lanes": int(B7_v_changing_lanes(df, "j")),
        "B8_i_passing": int(B8_v_passing(df, "i")),
        "B8_j_passing": int(B8_v_passing(df, "j")),
        
    }

    features.update(rule_features)

    return pd.DataFrame([features])



# Feature Engineering

def feature_eng(df):
    df1 = feature_eng_timestamp(df)
    df2 = feature_eng_pair(df1)
    return df2


# Rule-based Modeling

def rule_based_event(df):
    event = ""
    # ============================================================
    # CHECK ISSUE LANE
    # ============================================================
    if B0_unknowns(df):
        event = "NA" 
    # ============================================================
    # REAR-END CATEGORY
    # ============================================================
    elif B1_same_lane(df): # REAR-END
        if B2_v1_decelerating(df): # v1 is decelerating
            event = "26,15,r"
        else:
            event = "16,14,r"
    # ============================================================
    # CROSSING PATHS CATEGORY
    # ============================================================
    elif B3_crossing_paths(df): # CROSSING PATHS
        if B4_y_proximity(df): # v1 and v2 are in neighboring lanes
            if B5_v1_in_front(df): # v1 is the first to lane change
                if B6_v_from_right(df, "i"): # v1 coming from the right
                    event = "52,18,c"
                else: # v1 coming from left
                    event = "53,17,c"
            else: # v2 is the first to lane change
                if B6_v_from_right(df, "j"): # v2 coming from the right
                    event = "18,52,c"
                else: # v2 coming from left
                    event = "17,53,c"            
        else: # v1 and v2 are far, which in crossing indicate both are changing lanes
            if B5_v1_in_front(df): # v1 is the first to lane change
                if B7_v_changing_lanes(df, "j"):# check if v2 actually lane changes
                    if B6_v_from_right(df, "i"): # v1 coming from the right
                        event = "52,58,c"
                    else: # v1 coming from left
                        event = "53,57,c"
                else: # v2 does not lane changes
                    if B6_v_from_right(df, "i"): # v1 coming from the right
                        event = "52,18,c"
                    else: # v1 coming from left
                        event = "53,17,c"
            else: # v2 is the first to lane change
                if B7_v_changing_lanes(df, "i"): # check if v1 actually changes lanes
                    if B6_v_from_right(df, "j"): # v2 coming from the right
                        event = "58,52,c"
                    else: # v2 coming from left
                        event = "57,53,c"
                else: # v1 does not change lanes   
                    if B6_v_from_right(df, "j"): # v2 coming from the right
                        event = "18,52,c"
                    else: # v2 coming from left
                        event = "17,53,c"                    
    # ============================================================
    # LANE CHANGE CATEGORY
    # ============================================================

    elif B7_v_changing_lanes(df, "i") and B7_v_changing_lanes(df, "j"): # v1 and v2 are changing lanes
        if B5_v1_in_front(df): # v1 is the first to lane change
            if B6_v_from_right(df, "i"): # v1 coming from the right
                event = "52,58,l"
            else: # v1 coming from left
                event = "53,57,l"
        else: # v2 is the first to lane change
            if B6_v_from_right(df, "j"): # v2 coming from the right
                event = "58,52,l"
            else: # v2 coming from left
                event = "57,53,l"   
    elif B7_v_changing_lanes(df, "i") and not B7_v_changing_lanes(df, "j"): # v1 is cahnge lanes and v2 is not
        if not B4_y_proximity(df): # v2 is not changing lanes and is not in the neighboring lane of v1 (so no passing)
            if B6_v_from_right(df, "i"): # v1 coming from the right
                event = "52,18,l"
            else: # v1 coming from left
                event = "53,17,l"     
        else: # v1 is in the neighboring lane of v2
            if B8_v_passing(df, "j"): # v2 is passing v1
                if B6_v_from_right(df, "i"): # v1 coming from the right
                    event = "52,48,l"
                else: # v1 coming from left
                    event = "53,47,l"    
            else: # v2 is not passing v1
                if B6_v_from_right(df, "i"): # v1 coming from the right
                    event = "52,18,l"
                else: # v1 coming from left
                    event = "53,17,l"                      
    elif not B7_v_changing_lanes(df, "i") and B7_v_changing_lanes(df, "j"): # v2 is cahnge lanes and v1 is not
        if not B4_y_proximity(df): # v1 is not changing lanes and is not in the neighboring lane of v2 (so no passing)
            if B6_v_from_right(df, "j"): # v2 coming from the right
                event = "18,52,l"
            else: # v2 coming from left
                event = "17,53,l"  
        else: # v2 is in the neighboring lane of v1    
            if B8_v_passing(df, "i"): # v1 is passing v2
                if B6_v_from_right(df, "j"): # v2 coming from the right
                    event = "48,52,l"
                else: # v2 coming from left
                    event = "47,53,l"          
            else:
                if B6_v_from_right(df, "j"): # v2 coming from the right
                    event = "18,52,l"
                else: # v2 coming from left
                    event = "17,53,l"             

    # ============================================================
    # OTHER
    # ============================================================

    else:
        event = "NA"

    if (event[1] == "2" or event[1] == "3") and B13_y_and_heading_test(df):
        event = "NA"
    
    return event


def rule_based_steering(df, label):
    if label == "NA":
        return "NA"
    else:
        e1 = {"i":label[0],"j":label[3]}  # get event e1 of v1 and v2 to solve the case both no avoidance
        e2 = {"i":label[1],"j":label[4]}  # get event e2 of v1 and v2 to avoid conflict
        e3  = {"i":"?","j":"?"}
        for vehicle in ["i","j"]:
            steer = B9_test_steer(df, vehicle)
            # first adjsutments
            if ((e1[vehicle] == "5" and e2[vehicle] == "2") or (e1[vehicle] == "5" and e2[vehicle] == "7")) and (steer == "left"):
                steer = "no"
            if ((e1[vehicle] == "5" and e2[vehicle] == "3") or (e1[vehicle] == "5" and e2[vehicle] == "8")) and (steer == "right"):
                steer = "no"
        
            e3[vehicle] = steer
        event = e3["i"]+","+e3["j"]
    return event


def rule_based_acceleration(df, label):
    if label == "NA":
        return "NA"

    e3 = {"i": "?", "j": "?"}

    for vehicle in ["i", "j"]:
        e3[vehicle] = B10_test_acceleration_simple(df, vehicle)

    event = e3["i"] + "," + e3["j"]

    return event



# Learning-based Modeling

def clean_X(X):

    X = (
        X.replace([np.inf, -np.inf], np.nan)
         .fillna(0)
         .copy()
    )

    for col in X.select_dtypes(include="object"):

        X[col] = (
            X[col]
            .astype(str)
            .astype("category")
            .cat.codes
        )

    return X


def build_avoidance_label(
    steer,
    acceleration
):

    mapping = {

        ("no", "no"): "1",
        ("left", "no"): "2",
        ("right", "no"): "3",

        ("no", "dec"): "4",
        ("left", "dec"): "5",
        ("right", "dec"): "6",

        ("no", "acc"): "7",
        ("left", "acc"): "8",
        ("right", "acc"): "9",
    }

    return mapping[(steer, acceleration)]


# Generate Image

def vehicle_box(x, y, hx, hy, length, width):
    """
    Returns the 4 corners of a vehicle rectangle.

    Parameters
    ----------
    x, y : float
        Vehicle center position (ft).

    hx, hy : float
        Unit heading vector.

    length : float
        Vehicle length (ft).

    width : float
        Vehicle width (ft).
    """

    h = np.array([hx, hy], dtype=float)
    h /= np.linalg.norm(h)

    # Perpendicular vector
    p = np.array([-h[1], h[0]])

    half_l = length / 2.0
    half_w = width / 2.0

    corners = np.array([
        [x, y] + half_l * h + half_w * p,
        [x, y] + half_l * h - half_w * p,
        [x, y] - half_l * h - half_w * p,
        [x, y] - half_l * h + half_w * p,
    ])

    return corners


def plot_conflict(df,ids,t0_TTC,save_path=None):

    # ========================================================
    # FONT SIZES
    # ========================================================

    TITLE_SIZE = 18
    LABEL_SIZE = 18
    TICK_SIZE = 15
    LEGEND_SIZE = 15
    NUMBER_SIZE = 14

    # ========================================================
    # DATA RANGES
    # ========================================================

    xmin = min(df["x_i"].min(),df["x_j"].min())
    xmax = max(df["x_i"].max(),df["x_j"].max())
    ymin = min(df["y_i"].min(),df["y_j"].min(),-60)
    ymax = max(df["y_i"].max(),df["y_j"].max(),0)

    x_range = xmax - xmin
    y_range = ymax - ymin

    data_aspect = (
        y_range / x_range
        if x_range > 0
        else 1.0
    )

    # ========================================================
    # SETUP SUBPLOTS
    # ========================================================

    traj_height_weight = max(
        0.4,
        data_aspect * 4.0
    )

    fig, axs = plt.subplot_mosaic(
        [['traj'], ['speed'], ['acc']],
        figsize=(
            18,
            10 + (traj_height_weight * 2)
        ),
        gridspec_kw={
            'height_ratios': [
                traj_height_weight,
                1.0,
                1.0
            ]
        }
    )

    ax_traj = axs['traj']
    ax_speed = axs['speed']
    ax_acc = axs['acc']

    # Share X-axis
    ax_speed.sharex(ax_traj)
    ax_acc.sharex(ax_traj)

    # ========================================================
    # SPEED
    # ft/s -> mph
    # ========================================================

    speed_i = (
        np.sqrt(
            df["vx_i"]**2 +
            df["vy_i"]**2
        ) * 0.681818
    )

    speed_j = (
        np.sqrt(
            df["vx_j"]**2 +
            df["vy_j"]**2
        ) * 0.681818
    )

    # ========================================================
    # ACCELERATION
    # ft/s² -> g
    # ========================================================

    acc_i_g = df["acc_i"] / 32.174
    acc_j_g = df["acc_j"] / 32.174

    # ========================================================
    # SUBPLOT 1: TRAJECTORIES
    # ========================================================

    ax_traj.plot(
        df["x_i"].values,
        df["y_i"].values,
        color="black",
        linewidth=2,
        label="Veh i Trajectory"
    )

    ax_traj.plot(
        df["x_j"].values,
        df["y_j"].values,
        color="black",
        linewidth=2,
        label="Veh j Trajectory"
    )

    # --------------------------------------------------------
    # Light gray vehicle footprints
    # --------------------------------------------------------

    for _, row in df.iterrows():

        poly_i = Polygon(
            vehicle_box(
                row["x_i"],
                row["y_i"],
                row["hx_i"],
                row["hy_i"],
                row["length_i"],
                row["width_i"]
            ),
            facecolor="lightgray",
            edgecolor="none",
            alpha=0.10,
            zorder=1
        )

        poly_j = Polygon(
            vehicle_box(
                row["x_j"],
                row["y_j"],
                row["hx_j"],
                row["hy_j"],
                row["length_j"],
                row["width_j"]
            ),
            facecolor="lightgray",
            edgecolor="none",
            alpha=0.10,
            zorder=1
        )

        ax_traj.add_patch(poly_i)
        ax_traj.add_patch(poly_j)

    # ========================================================
    # VEHICLE POSITIONS EVERY 1 SECOND
    # Assumes 25 frames/second
    # ========================================================

    total_len = len(df)

    periodic_indices = list(
        range(0, total_len, 25)
    )

    if (total_len - 1) not in periodic_indices:
        periodic_indices.append(
            total_len - 1
        )

    for order_idx, idx in enumerate(
        periodic_indices
    ):

        row = df.iloc[idx]

        # Vehicle i
        poly_i = Polygon(
            vehicle_box(
                row["x_i"],
                row["y_i"],
                row["hx_i"],
                row["hy_i"],
                row["length_i"],
                row["width_i"]
            ),
            facecolor="blue",
            edgecolor="k",
            linewidth=1.0,
            alpha=0.8,
            zorder=11
        )

        # Vehicle j
        poly_j = Polygon(
            vehicle_box(
                row["x_j"],
                row["y_j"],
                row["hx_j"],
                row["hy_j"],
                row["length_j"],
                row["width_j"]
            ),
            facecolor="orange",
            edgecolor="k",
            linewidth=1.0,
            alpha=0.8,
            zorder=11
        )

        ax_traj.add_patch(poly_i)
        ax_traj.add_patch(poly_j)

        # ----------------------------------------------------
        # Vehicle i timestamp number
        # ----------------------------------------------------

        txt = ax_traj.text(
            row["x_i"],
            row["y_i"],
            str(order_idx),
            color="white",
            fontsize=NUMBER_SIZE,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=50
        )

        txt.set_path_effects([
            pe.Stroke(
                linewidth=3.5,
                foreground="black"
            ),
            pe.Normal()
        ])

        # ----------------------------------------------------
        # Vehicle j timestamp number
        # ----------------------------------------------------

        txt = ax_traj.text(
            row["x_j"],
            row["y_j"],
            str(order_idx),
            color="white",
            fontsize=NUMBER_SIZE,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=50
        )

        txt.set_path_effects([
            pe.Stroke(
                linewidth=3.5,
                foreground="black"
            ),
            pe.Normal()
        ])

    # ========================================================
    # HIGHLIGHT MINIMUM TTC FRAME
    # ========================================================

    if t0_TTC is not None:

        # Find row whose t is closest to t0_TTC
        ttc_idx = np.argmin(
            np.abs(
                df["t"] - t0_TTC
            )
        )

        row_ttc = df.iloc[ttc_idx]

        # Vehicle i at minimum TTC
        poly_i_ttc = Polygon(
            vehicle_box(
                row_ttc["x_i"],
                row_ttc["y_i"],
                row_ttc["hx_i"],
                row_ttc["hy_i"],
                row_ttc["length_i"],
                row_ttc["width_i"]
            ),
            facecolor="red",
            edgecolor="k",
            linewidth=1.5,
            alpha=0.9,
            zorder=15
        )

        # Vehicle j at minimum TTC
        poly_j_ttc = Polygon(
            vehicle_box(
                row_ttc["x_j"],
                row_ttc["y_j"],
                row_ttc["hx_j"],
                row_ttc["hy_j"],
                row_ttc["length_j"],
                row_ttc["width_j"]
            ),
            facecolor="red",
            edgecolor="k",
            linewidth=1.5,
            alpha=0.9,
            zorder=15
        )

        ax_traj.add_patch(poly_i_ttc)
        ax_traj.add_patch(poly_j_ttc)

    # ========================================================
    # ROAD LANES
    # Direction is always 1
    # ========================================================

    y_edges = [
        -12,
        -24,
        -36,
        -48,
        -60
    ]

    for y in y_edges:

        if y in [-12, -60]:

            ax_traj.plot(
                [xmin, xmax],
                [y, y],
                "-",
                color="gray",
                linewidth=1,
                alpha=0.6
            )

        else:

            ax_traj.plot(
                [xmin, xmax],
                [y, y],
                "--",
                color="gray",
                linewidth=1,
                alpha=0.6
            )

    ax_traj.grid(
        True,
        alpha=0.3
    )

    # ========================================================
    # SUBPLOT 2: SPEED PROFILE
    # ========================================================

    ax_speed.plot(
        df["x_i"].values,
        speed_i.values,
        color="blue",
        linewidth=2,
        label="Vehicle 1"
    )

    ax_speed.plot(
        df["x_j"].values,
        speed_j.values,
        color="orange",
        linewidth=2,
        label="Vehicle 2"
    )

    ax_speed.grid(
        True,
        alpha=0.3
    )

    ax_speed.legend(
        loc="upper right",
        fontsize=LEGEND_SIZE
    )

    v_min_actual = min(
        speed_i.min(),
        speed_j.min()
    )

    v_max_actual = max(
        speed_i.max(),
        speed_j.max()
    )

    if (
        v_min_actual >= 40
        and v_max_actual <= 90
    ):

        ax_speed.set_ylim(
            40,
            90
        )

    else:

        ax_speed.set_ylim(
            max(
                0,
                v_min_actual - 5
            ),
            v_max_actual + 5
        )

    # ========================================================
    # SUBPLOT 3: ACCELERATION PROFILE
    # ========================================================

    ax_acc.plot(
        df["x_i"].values,
        acc_i_g.values,
        color="blue",
        linewidth=2
    )

    ax_acc.plot(
        df["x_j"].values,
        acc_j_g.values,
        color="orange",
        linewidth=2
    )

    # Hard braking thresholds
    ax_acc.axhline(
        y=0.25,
        color="green",
        linestyle=":",
        linewidth=2,
        alpha=0.8
    )

    ax_acc.axhline(
        y=-0.25,
        color="green",
        linestyle=":",
        linewidth=2,
        alpha=0.8,
        label="Hard Braking - Lower End"
    )

    ax_acc.axhline(
        y=0.78,
        color="red",
        linestyle=":",
        linewidth=2,
        alpha=0.8
    )

    ax_acc.axhline(
        y=-0.78,
        color="red",
        linestyle=":",
        linewidth=2,
        alpha=0.8,
        label="Hard Braking - Higher End"
    )

    ax_acc.set_ylim(
        -0.8,
        0.8
    )

    ax_acc.set_yticks([
        -0.78,
        -0.50,
        -0.25,
        0,
        0.25,
        0.50,
        0.78
    ])

    ax_acc.set_yticklabels([
        "-0.78",
        "-0.50",
        "-0.25",
        "0",
        "0.25",
        "0.50",
        "0.78"
    ])

    ax_acc.grid(
        True,
        alpha=0.3
    )

    ax_acc.legend(
        loc="upper right",
        fontsize=LEGEND_SIZE
    )

    # ========================================================
    # TRAJECTORY AXIS
    # ========================================================

    ax_traj.set_xlim(
        xmin,
        xmax
    )

    ax_traj.set_ylim(
        ymin - 5,
        ymax + 5
    )

    ax_traj.set_aspect(
        "equal",
        adjustable="datalim",
        anchor="C"
    )


    # ========================================================
    # X-AXIS FORMAT
    # ========================================================

    mile_formatter = FuncFormatter(
        lambda x, pos: f"{x / 5280:.2f}"
    )

    ax_speed.xaxis.set_major_formatter(
        mile_formatter
    )

    ax_acc.xaxis.set_major_formatter(
        mile_formatter
    )

    # ========================================================
    # LABEL STYLING
    # ========================================================

    for ax in [
        ax_traj,
        ax_speed,
        ax_acc
    ]:

        ax.tick_params(
            axis="both",
            labelsize=TICK_SIZE
        )

    ax_traj.set_ylabel(
        "Y (ft)",
        fontsize=LABEL_SIZE
    )

    ax_speed.set_ylabel(
        "Speed (mph)",
        fontsize=LABEL_SIZE
    )

    ax_acc.set_xlabel(
        "X Position (mile)",
        fontsize=LABEL_SIZE
    )

    ax_acc.set_ylabel(
        "Acceleration (g)",
        fontsize=LABEL_SIZE
    )

    plt.setp(
        ax_traj.get_xticklabels(),
        visible=False
    )

    plt.setp(
        ax_speed.get_xticklabels(),
        visible=False
    )

    # ========================================================
    # TITLE
    # ========================================================

    id1, id2 = ids.split(
        "_",
        1
    )

    title_string = (
        f"Vehicle 1 ID: {id1}    "
        f"Vehicle 2 ID: {id2}"
    )

    ax_traj.set_title(
        title_string,
        fontsize=TITLE_SIZE,
        pad=28
    )

    # ========================================================
    # SAVE
    # ========================================================

    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=100,
            bbox_inches="tight"
        )

    return fig, (
        ax_traj,
        ax_speed,
        ax_acc
    )