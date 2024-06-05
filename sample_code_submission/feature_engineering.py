import pandas as pd
import numpy as np
import time


def feature_engineering(df, nan_to_minus_7=True):
    """
    Perform feature engineering operations on the input dataframe
    and create a new dataframe with only the features required for training the model.
    """

    # Measure time of execution
    start = time.time()

    # Perform calculations to derive features from the DataFrame
    # and store the results in new columns

    # Engineered features to be used for calculations

    lep_px = df["PRI_lep_pt"] * np.cos(df["PRI_lep_phi"])
    lep_py = df["PRI_lep_pt"] * np.sin(df["PRI_lep_phi"])
    lep_pz = df["PRI_lep_pt"] * np.sinh(df["PRI_lep_eta"])

    had_px = df["PRI_had_pt"] * np.cos(df["PRI_had_phi"])
    had_py = df["PRI_had_pt"] * np.sin(df["PRI_had_phi"])
    had_pz = df["PRI_had_pt"] * np.sinh(df["PRI_had_eta"])

    met_x = df["PRI_met"] * np.cos(df["PRI_met_phi"])
    met_y = df["PRI_met"] * np.sin(df["PRI_met_phi"])

    jet_leading_px = df["PRI_jet_leading_pt"] * np.cos(df["PRI_jet_leading_phi"])
    jet_leading_py = df["PRI_jet_leading_pt"] * np.sin(df["PRI_jet_leading_phi"])
    jet_leading_pz = df["PRI_jet_leading_pt"] * np.sinh(df["PRI_jet_leading_eta"])

    jet_subleading_px = df["PRI_jet_subleading_pt"] * np.cos(df["PRI_jet_subleading_phi"])
    jet_subleading_py = df["PRI_jet_subleading_pt"] * np.sin(df["PRI_jet_subleading_phi"])
    jet_subleading_pz = df["PRI_jet_subleading_pt"] * np.sinh(df["PRI_jet_subleading_eta"])

    # Put -7 to NaN
    for col in df:
        df[col][df[col] == -7] = np.nan

    # Correct PRI_n_jets
    n_jets = np.where(
        np.isfinite(df["PRI_jet_leading_pt"]),
        np.where(
            np.isfinite(df["PRI_jet_subleading_pt"]),
            # if PRI_leading_jet_pt != -7 and PRI_subleading_jet_pt != -7 => PRI_n_jets = 2
            np.full_like(df["PRI_n_jets"], 2),
            # if PRI_leading_jet_pt != -7 and PRI_subleading_jet_pt == -7 => PRI_n_jets = 1
            np.full_like(df["PRI_n_jets"], 1),
        ),
        # if PRI_leading_jet_pt == -7 and PRI_leading_jet_pt == -7 => PRI_n_jets = 0
        np.full_like(df["PRI_n_jets"], 0),
    )
    df["PRI_n_jets"] = n_jets

    # Engineered features to be passed on to the dataframe

    # DER_mass_vis, formule 21
    df["DER_mass_vis"] = np.sqrt(
        (np.sqrt(lep_px**2 + lep_py**2 + lep_pz**2) + np.sqrt(had_px**2 + had_py**2 + had_pz**2))
        ** 2
        - (lep_px + had_px) ** 2
        - (lep_py + had_py) ** 2
        - (lep_pz + had_pz) ** 2
    )

    # DER_pt_h, formule 20
    df["DER_pt_h"] = np.sqrt((lep_px + had_px + met_x) ** 2 + (lep_py + had_py + met_y) ** 2)

    # DER_mass_transverse_met_lep, formule 22
    df["DER_mass_transverse_met_lep"] = np.sqrt(
        (df["PRI_lep_pt"] + df["PRI_met"]) ** 2 - (lep_px + met_x) ** 2 - (lep_py + met_y) ** 2
    )

    # DER_mass_jet_jet, formule 21
    df["DER_mass_jet_jet"] = np.where(
        n_jets >= 2,
        np.sqrt(
            (
                np.sqrt(jet_leading_px**2 + jet_leading_py**2 + jet_leading_pz**2)
                + np.sqrt(jet_subleading_px**2 + jet_subleading_py**2 + jet_subleading_pz**2)
            )
            ** 2
            - (jet_leading_px + jet_subleading_px) ** 2
            - (jet_leading_py + jet_subleading_py) ** 2
            - (jet_leading_pz + jet_subleading_pz) ** 2
        ),
        np.nan,
    )

    # DER_deltaeta_jet_jet, formule 23
    df["DER_deltaeta_jet_jet"] = np.where(
        n_jets >= 2, np.abs(df["PRI_jet_leading_eta"] - df["PRI_jet_subleading_eta"]), np.nan
    )

    # DER_sum_pt
    df["DER_sum_pt"] = np.where(
        n_jets == 0,
        df["PRI_lep_pt"] + df["PRI_had_pt"],
        np.where(
            n_jets == 1,
            df["PRI_lep_pt"] + df["PRI_had_pt"] + df["PRI_jet_leading_pt"],
            np.where(
                n_jets == 2,
                df["PRI_lep_pt"]
                + df["PRI_had_pt"]
                + df["PRI_jet_leading_pt"]
                + df["PRI_jet_subleading_pt"],
                np.where(
                    n_jets == 3, df["PRI_lep_pt"] + df["PRI_had_pt"] + df["PRI_jet_all_pt"], np.nan
                ),
            ),
        ),
    )

    # DER_met_phi_centrality
    A = np.sin(df["PRI_met_phi"] - df["PRI_lep_phi"]) * np.sign(
        np.sin(df["PRI_had_phi"] - df["PRI_lep_phi"])
    )
    B = np.sin(df["PRI_had_phi"] - df["PRI_met_phi"]) * np.sign(
        np.sin(df["PRI_had_phi"] - df["PRI_lep_phi"])
    )
    df["DER_met_phi_centrality"] = (A + B) / np.sqrt(A**2 + B**2)

    # DER_prodeta_jet_jet
    df["DER_prodeta_jet_jet"] = np.where(
        n_jets >= 2, df["PRI_jet_leading_eta"] * df["PRI_jet_subleading_eta"], np.nan
    )

    # DER_deltar_tau_lep
    df["DER_deltar_tau_lep"] = np.sqrt(
        (df["PRI_lep_phi"] - df["PRI_had_phi"]) ** 2 + (df["PRI_lep_eta"] - df["PRI_had_eta"]) ** 2
    )

    # DER_lep_eta_centrality
    df["DER_lep_eta_centrality"] = np.where(
        n_jets >= 2,
        np.exp(
            (
                -4
                * (
                    df["PRI_lep_eta"]
                    - (df["PRI_jet_leading_eta"] + df["PRI_jet_subleading_eta"]) / 2
                )
                ** 2
            )
            / (df["PRI_jet_leading_eta"] - df["PRI_jet_subleading_eta"]) ** 2
        ),
        np.nan,
    )

    # DER_pt_ratio_lep_tau
    df["DER_pt_ratio_lep_tau"] = df["PRI_lep_pt"] / df["PRI_had_pt"]

    # DER_pt_tot
    df["DER_pt_tot"] = np.where(
        n_jets == 2,
        np.sqrt(
            lep_pz**2
            + had_pz**2
            + jet_leading_pz**2
            + jet_subleading_pz**2
            + df["PRI_had_pt"] ** 2
            + df["PRI_lep_pt"] ** 2
            + df["PRI_jet_leading_pt"] ** 2
            + df["PRI_jet_subleading_pt"] ** 2
        ),
        np.where(
            n_jets >= 1,
            np.sqrt(
                lep_pz**2
                + had_pz**2
                + jet_leading_pz**2
                + df["PRI_had_pt"] ** 2
                + df["PRI_lep_pt"] ** 2
                + df["PRI_jet_leading_pt"] ** 2
            ),
            np.nan,
        ),
    )

    # Put NaN back to -7 if needed
    if nan_to_minus_7:
        for col in df:
            df[col][np.logical_not(np.isfinite(df[col]))] = -7

    # Create the new dataframe
    new_columns = df.columns.tolist()
    df_new = pd.DataFrame(df, columns=new_columns)

    end = time.time()
    print(f"feature engineering took {end - start:.2f} s")

    return df_new