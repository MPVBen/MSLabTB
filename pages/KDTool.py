# titrage_ms_app_Hill_Gabelica.py

# Streamlit app for KD determination using Hill model and Gabelica method

# Modified to include Gabelica method alongside Hill, with equimolar titration option

# CORRECTED VERSION: Gabelica method now fits I(A)/I(AB) with Ka according to original paper

# LATEST FIXES:

# - Fixed indentation error in trapezoid import section

# - Prevent asymptote problem by starting fit curves from first measured value

# - Improved Gabelica fitting with wider bounds for R parameter and better diagnostics

# - Added outlier exclusion functionality with session state management (FIXED initialization order)

import io

import warnings

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import streamlit as st

from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# Initialize session state at the very beginning
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'avg_df' not in st.session_state:
    st.session_state.avg_df = None

# -----------------------------------------------------------------------------

# Integration helper: try to import a trapezoidal integration function

# -----------------------------------------------------------------------------

try:
    from numpy import trapezoid as trapz_func # NumPy ≥ 1.21
except Exception:
    try:
        from numpy import trapz as trapz_func # Fallback: np.trapz
    except Exception:
        try:
            from scipy.integrate import trapezoid as trapz_func # SciPy ≥ 1.7
        except Exception:
            from scipy.integrate import trapz as trapz_func # Final fallback

# -----------------------------------------------------------------------------

# I/O — data preparation

# -----------------------------------------------------------------------------

def load_csv_data(uploaded_file):
    """Parse the CSV produced by the native MS titration workflow.
    Expected layout:
    line 1: blank, replicate numbers (every second column)
    line 2: blank, ligand concentrations µM (every second column)
    line 3: blank
    line 4+: m/z, intensity, m/z, intensity, ... for each experiment
    Returns a list of dicts with keys: replicat, concentration, data (DataFrame).
    """
    try:
        df = pd.read_csv(uploaded_file, header=None)
        replicat_info = df.iloc[0, 1::2].dropna().astype(int).tolist()
        concentration_info = df.iloc[1, 1::2].dropna().astype(float).tolist()
        ms_data = df.iloc[3:].reset_index(drop=True)
        experiments = []
        col_pairs = [(i, i + 1) for i in range(0, len(ms_data.columns), 2)]
        for idx, (mz_col, int_col) in enumerate(col_pairs):
            if idx < len(concentration_info):
                exp_data = ms_data[[mz_col, int_col]].dropna()
                exp_data.columns = ["MZ", "Intensity"]
                experiments.append({
                    "replicat": replicat_info[idx] if idx < len(replicat_info) else 1,
                    "concentration": concentration_info[idx],
                    "data": exp_data.astype(float),
                })
        return experiments
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {str(e)}")
        return None

# -----------------------------------------------------------------------------

# Plot helpers

# -----------------------------------------------------------------------------

def plot_mass_spectra(experiments, selected_concentrations=None):
    if not experiments:
        return None
    # Keep one experiment per concentration (first replicate)
    conc_data = {}
    for exp in experiments:
        conc = exp["concentration"]
        if conc not in conc_data:
            conc_data[conc] = exp
    if selected_concentrations is not None and len(selected_concentrations) > 0:
        conc_data = {k: v for k, v in conc_data.items() if k in selected_concentrations}
    fig = go.Figure()
    colors = px.colors.qualitative.Set1
    for idx, (conc, exp) in enumerate(sorted(conc_data.items())):
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=exp["data"]["MZ"],
                y=exp["data"]["Intensity"],
                mode="lines",
                name=f"Ligand {conc} µM",
                line=dict(color=color),
            )
        )
    fig.update_layout(
        title="Mass spectra by ligand concentration",
        xaxis_title="m/z",
        yaxis_title="Intensity",
        hovermode="x unified",
        showlegend=True,
    )
    return fig

def integrate_peaks(data, mz_ranges):
    """Integrate Intensity vs. m/z over a list of (min, max) ranges."""
    total_area = 0.0
    for mz_min, mz_max in mz_ranges:
        mask = (data["MZ"] >= mz_min) & (data["MZ"] <= mz_max)
        filtered = data[mask].sort_values("MZ")
        if len(filtered) > 1:
            total_area += trapz_func(filtered["Intensity"], filtered["MZ"])
    return total_area

# -----------------------------------------------------------------------------

# Binding models

# -----------------------------------------------------------------------------

def hill_binding_model(L0, Kd_uM, n, Rmax):
    """Hill model: R = Rmax * L^n / (Kd^n + L^n)"""
    L0 = np.asarray(L0, dtype=float)
    numerator = Rmax * (L0 ** n)
    denominator = (Kd_uM ** n) + (L0 ** n)
    return np.where(denominator > 0, numerator / denominator, 0.0)

def hill_binding_model_fixed_n(L0, Kd_uM, Rmax, n_fixed=1.0):
    """Hill model with fixed n: R = Rmax * L^n / (Kd^n + L^n)"""
    L0 = np.asarray(L0, dtype=float)
    numerator = Rmax * (L0 ** n_fixed)
    denominator = (Kd_uM ** n_fixed) + (L0 ** n_fixed)
    return np.where(denominator > 0, numerator / denominator, 0.0)

def gabelica_binding_model_corrected(C0, Ka, R):
    """
    Gabelica method for equimolar titration according to equation 11 from Gabelica et al. (2003):
    I(A)/I(AB) = (1 + (1 + 4*Ka*C0)^(1/2)) / (2*R*Ka*C0)
    C0: concentrations (equimolar) in µM
    Ka: association constant in µM^-1
    R: response factor ratio (RAB/RA)
    Returns: I(A)/I(AB) ratio
    """
    C0 = np.asarray(C0, dtype=float)
    Ka = max(Ka, 1e-10) # Avoid division by zero
    R = max(R, 1e-10) # Avoid division by zero
    # Equation 11: I(A)/I(AB) = (1 + (1 + 4*Ka*C0)^(1/2)) / (2*R*Ka*C0)
    sqrt_term = np.sqrt(1 + 4*Ka*C0)
    I_A_over_I_AB = (1 + sqrt_term) / (2 * R * Ka * C0)
    return I_A_over_I_AB

# -----------------------------------------------------------------------------

# Fit helpers

# -----------------------------------------------------------------------------

def compute_r2(y_obs, y_pred):
    resid = y_obs - y_pred
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot

def compute_aic_bic(y_obs, y_pred, k):
    n = len(y_obs)
    residuals = y_obs - y_pred
    ss_res = np.sum(residuals ** 2)
    sigma2 = ss_res / max(n, 1)
    sigma2 = max(sigma2, 1e-12)
    aic = n * np.log(sigma2) + 2.0 * k
    bic = n * np.log(sigma2) + k * np.log(max(n, 1))
    return aic, bic

def fit_hill(L0, ratio, ratio_err=None, use_weights=False, fix_n_to_1=False):
    L0 = np.asarray(L0, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    valid = (L0 >= 0) & np.isfinite(L0) & np.isfinite(ratio)
    x, y = L0[valid], ratio[valid]
    if len(x) < 2: # Reduced from 3 to 2 when n is fixed
        return None
    sigma = ratio_err[valid] if (use_weights and ratio_err is not None) else None
    try:
        if fix_n_to_1:
            # Fixed n=1 model: only fit Kd and Rmax
            p0 = [np.median(x) if np.median(x) > 0 else 1.0, max(np.max(y), 1e-3)]
            bounds = ([1e-5, 1e-6], [1e6, 1e6])
            def model_fixed(L, Kd_uM, Rmax):
                return hill_binding_model_fixed_n(L, Kd_uM, Rmax, n_fixed=1.0)
            popt, pcov = curve_fit(
                model_fixed,
                x,
                y,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=use_weights,
                maxfev=20000,
            )
            # Reconstruct full parameter array with n=1
            popt_full = [popt[0], 1.0, popt[1]] # [Kd, n=1, Rmax]
            if pcov is not None:
                perr = np.sqrt(np.diag(pcov))
                perr_full = [perr[0], 0.0, perr[1]] # [Kd_err, n_err=0, Rmax_err]
            else:
                perr_full = np.full(3, np.nan)
            ypred = model_fixed(x, *popt)
            ypred_full_func = lambda L: hill_binding_model_fixed_n(L, popt[0], popt[1], n_fixed=1.0)
            n_params = 2
        else:
            # Variable n model: fit all three parameters
            if len(x) < 3:
                return None
            p0 = [np.median(x) if np.median(x) > 0 else 1.0, 1.0, max(np.max(y), 1e-3)]
            bounds = ([1e-5, 0.1, 1e-6], [1e6, 5.0, 1e6])
            popt, pcov = curve_fit(
                hill_binding_model,
                x,
                y,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=use_weights,
                maxfev=20000,
            )
            popt_full = popt
            perr_full = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(3, np.nan)
            ypred = hill_binding_model(x, *popt)
            ypred_full_func = lambda L: hill_binding_model(L, *popt)
            n_params = 3
        r2 = compute_r2(y, ypred)
        aic, bic = compute_aic_bic(y, ypred, k=n_params)
        return {
            "popt": popt_full,
            "perr": perr_full,
            "r2": r2,
            "aic": aic,
            "bic": bic,
            "ypred_full": ypred_full_func,
            "fixed_n": fix_n_to_1,
        }
    except Exception as e:
        st.error(f"Erreur lors du fit Hill: {str(e)}")
        return None

def fit_gabelica_corrected(C0, ratio_inverse, ratio_err=None, use_weights=False):
    """
    Fit the corrected Gabelica model for equimolar titration using equation 11
    C0: concentrations (equimolar)
    ratio_inverse: I(A)/I(AB) ratios (inverse of the previous implementation)
    IMPROVED VERSION: Wider bounds for R and better initial parameter estimation
    """
    C0 = np.asarray(C0, dtype=float)
    ratio_inverse = np.asarray(ratio_inverse, dtype=float)
    valid = (C0 > 0) & np.isfinite(C0) & np.isfinite(ratio_inverse) & (ratio_inverse > 0)
    x, y = C0[valid], ratio_inverse[valid]
    if len(x) < 2:
        return None
    sigma = ratio_err[valid] if (use_weights and ratio_err is not None) else None
    try:
        # IMPROVED initial parameter guesses
        Ka_guess = 1.0 / np.median(x) if np.median(x) > 0 else 1.0
        # Better R initial guess: estimate from data trend
        if len(y) >= 2:
            # Estimate R from the ratio of high vs low concentration responses
            y_sorted_idx = np.argsort(x)
            x_sorted, y_sorted = x[y_sorted_idx], y[y_sorted_idx]
            # Take first 30% and last 30% of data
            n_points = len(y_sorted)
            n_low = max(1, int(0.3 * n_points))
            n_high = max(1, int(0.3 * n_points))
            y_low_avg = np.mean(y_sorted[:n_low])
            y_high_avg = np.mean(y_sorted[-n_high:])
            # R initial guess based on ratio behavior
            if y_high_avg > 0 and y_low_avg > 0:
                R_guess = min(max(y_low_avg / y_high_avg, 0.01), 1000.0)
            else:
                R_guess = 1.0
        else:
            R_guess = 1.0
        p0 = [Ka_guess, R_guess]
        # EXPANDED bounds: much wider range for R parameter
        # Ka: very wide range (1e-10 to 1e10 µM^-1)
        # R: much wider range (0.00000001 to 100000000) to avoid hitting limits
        bounds = ([1e-10, 0.00000001], [1e10, 100000000.0])
        popt, pcov = curve_fit(
            gabelica_binding_model_corrected,
            x,
            y,
            p0=p0,
            bounds=bounds,
            sigma=sigma,
            absolute_sigma=use_weights,
            maxfev=50000, # Increased max iterations
        )
        Ka_fit, R_fit = popt
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(2, np.nan)
        ypred = gabelica_binding_model_corrected(x, *popt)
        ypred_full_func = lambda C: gabelica_binding_model_corrected(C, *popt)
        r2 = compute_r2(y, ypred)
        aic, bic = compute_aic_bic(y, ypred, k=2)
        # Convert Ka to Kd
        Kd_fit = 1.0 / Ka_fit
        Kd_err = perr[0] / (Ka_fit**2) if not np.isnan(perr[0]) else np.nan
        # Check if parameters hit bounds (diagnostic)
        Ka_min, R_min = bounds[0]
        Ka_max, R_max = bounds[1]
        bounds_warning = []
        if abs(Ka_fit - Ka_min) / Ka_min < 0.01:
            bounds_warning.append("Ka proche de la limite inférieure")
        if abs(Ka_fit - Ka_max) / Ka_max < 0.01:
            bounds_warning.append("Ka proche de la limite supérieure")
        if abs(R_fit - R_min) / R_min < 0.01:
            bounds_warning.append("R proche de la limite inférieure")
        if abs(R_fit - R_max) / R_max < 0.01:
            bounds_warning.append("R proche de la limite supérieure")
        return {
            "popt": [Kd_fit, R_fit], # Return Kd instead of Ka for consistency with UI
            "popt_original": popt, # Keep original Ka, R for internal use
            "perr": [Kd_err, perr[1]],
            "perr_original": perr,
            "r2": r2,
            "aic": aic,
            "bic": bic,
            "ypred_full": ypred_full_func,
            "Ka": Ka_fit,
            "Kd": Kd_fit,
            "R": R_fit,
            "bounds_warning": bounds_warning, # NEW: diagnostic info
            "initial_guess": [Ka_guess, R_guess], # NEW: for debugging
        }
    except Exception as e:
        st.error(f"Erreur lors du fit Gabelica: {str(e)}")
        return None

# -----------------------------------------------------------------------------

# Residuals plot

# -----------------------------------------------------------------------------

def make_residuals_plot(L0, ratio, hill_res, gabelica_res=None):
    fig = go.Figure()
    if hill_res is not None:
        res_h = ratio - hill_res["ypred_full"](L0)
        fig.add_trace(
            go.Bar(x=L0, y=res_h, name="Hill residuals", marker_color="red", opacity=0.7)
        )
    if gabelica_res is not None:
        res_g = ratio - gabelica_res["ypred_full"](L0)
        fig.add_trace(
            go.Bar(x=L0, y=res_g, name="Gabelica residuals", marker_color="blue", opacity=0.7, offsetgroup=2)
        )
    fig.update_layout(
        title="Résidus (données − modèle)",
        xaxis_title="Concentration (µM)",
        yaxis_title="Résidu",
        width=900,
        height=400,
        barmode='group'
    )
    return fig

# -----------------------------------------------------------------------------

# Streamlit UI

# -----------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="KD determination: Hill & Gabelica methods",
        layout="wide",
        page_icon="🧪",
    )
    st.title("Détermination du KD — Modèles de Hill et Gabelica")
    
    # Sidebar — I/O and options
    st.sidebar.header("Paramètres")
    uploaded_file = st.sidebar.file_uploader("Choisir le fichier CSV", type=["csv"])

    # Titration type selection
    st.sidebar.subheader("Type de titrage")
    equimolar_titration = st.sidebar.checkbox(
        "Titrage équimolaire (cible = ligand)",
        value=False,
        help="Cochez si les concentrations de cible et ligand sont égales (nécessaire pour méthode Gabelica)"
    )

    # Method selection
    st.sidebar.subheader("Méthodes d'ajustement")
    use_hill = st.sidebar.checkbox("Modèle de Hill", value=True)
    use_gabelica = st.sidebar.checkbox(
        "Méthode de Gabelica (améliorée)",
        value=False,
        help="Méthode de Gabelica et al. (2003) - fitte I(A)/I(AB) avec Ka - nécessite un titrage équimolaire - AMÉLIORÉ avec limites élargies pour R"
    )

    if use_gabelica and not equimolar_titration:
        st.sidebar.warning("⚠️ La méthode de Gabelica nécessite un titrage équimolaire")
        use_gabelica = False

    # Hill model options
    if use_hill:
        st.sidebar.subheader("Options du modèle de Hill")
        fix_n_to_1 = st.sidebar.checkbox(
            "Fixer n = 1 (pas de coopérativité)",
            value=False,
            help="Cochez cette case si vous êtes sûr qu'il n'y a pas de phénomène de coopérativité"
        )
    else:
        fix_n_to_1 = False
    )

    # Outlier exclusion widget - only show after analysis is done
    removed_conc = []
    if st.session_state.analysis_done and st.session_state.avg_df is not None:
        st.sidebar.subheader("⚠️ Exclusion d'outliers")
        all_conc = st.session_state.avg_df["concentration"].tolist()
        removed_conc = st.sidebar.multiselect(
            "Concentrations à exclure du fit (outliers)", all_conc, default=[],
            help="Sélectionnez les concentrations à exclure de l'ajustement si vous suspectez qu'il s'agit d'outliers",
            key="outlier_exclusion"
        )

    # Add reset button in sidebar
    if st.session_state.analysis_done:
        if st.sidebar.button("🔄 Réinitialiser l'analyse"):
            st.session_state.analysis_done = False
            st.session_state.avg_df = None
            st.rerun()

    # --------------------------------------------------------------------------------
    # Main logic once file is provided
    # --------------------------------------------------------------------------------

    if uploaded_file is not None:
        with st.spinner("Chargement des données..."):
            experiments = load_csv_data(uploaded_file)

        if experiments:
            # ---------------------------------------------------------------------
            # Show data summary and spectra
            # ---------------------------------------------------------------------
            st.subheader("Résumé des données")
            info_df = pd.DataFrame([
                {
                    "réplica": exp["replicat"],
                    "concentration (µM)": exp["concentration"],
                    "n_points": len(exp["data"]),
                }
                for exp in experiments
            ])
            st.dataframe(info_df)

            concentrations = sorted({exp["concentration"] for exp in experiments})
            default_sel = concentrations[: min(6, len(concentrations))]
            sel_conc = st.sidebar.multiselect(
                "Concentrations à afficher", concentrations, default=default_sel
            )

            st.subheader("Spectres de masse")
            fig_spec = plot_mass_spectra(experiments, selected_concentrations=sel_conc)
            if fig_spec:
                st.plotly_chart(fig_spec, use_container_width=True)

            # ------------------------------------------------------------------
            # Define m/z ranges
            # ------------------------------------------------------------------
            st.subheader("Définir les plages m/z pour l'intégration")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Pics de la cible libre (A)")
                n_rec = st.number_input("Nombre de plages cible", 1, 5, 1, key="nrec")
                receptor_ranges = []
                for i in range(n_rec):
                    r1, r2 = st.columns(2)
                    with r1:
                        mn = st.number_input(f"A min {i+1}", value=100.0 + 50 * i, key=f"recmin{i}")
                    with r2:
                        mx = st.number_input(f"A max {i+1}", value=120.0 + 50 * i, key=f"recmax{i}")
                    if mn < mx:
                        receptor_ranges.append((mn, mx))

            with col2:
                st.write("Pics du complexe (AB)")
                n_comp = st.number_input("Nombre de plages complexe", 1, 5, 1, key="ncomp")
                complex_ranges = []
                for i in range(n_comp):
                    c1, c2 = st.columns(2)
                    with c1:
                        mn = st.number_input(f"AB min {i+1}", value=300.0 + 50 * i, key=f"plmin{i}")
                    with c2:
                        mx = st.number_input(f"AB max {i+1}", value=320.0 + 50 * i, key=f"plmax{i}")
                    if mn < mx:
                        complex_ranges.append((mn, mx))

            # ------------------------------------------------------------------
            # Run analysis - SEPARATE INTEGRATION FROM FITTING
            # ------------------------------------------------------------------
            if st.button("Lancer l'analyse") or st.session_state.analysis_done:
                if not receptor_ranges or not complex_ranges:
                    st.error("Définir au moins une plage pour la cible et une pour le complexe.")
                    return

                # Only do integration if not already done or if button is pressed
                if not st.session_state.analysis_done:
                    with st.spinner("Intégration des pics et moyenne des réplicats..."):
                        results = []
                        for exp in experiments:
                            rec_area = integrate_peaks(exp["data"], receptor_ranges)
                            comp_area = integrate_peaks(exp["data"], complex_ranges)
                            # For Hill: ratio = I(AB)/I(A) = comp_area / rec_area
                            ratio_hill = comp_area / rec_area if rec_area > 0 else 0.0
                            # For Gabelica: ratio_inverse = I(A)/I(AB) = rec_area / comp_area
                            ratio_gabelica = rec_area / comp_area if comp_area > 0 else np.inf
                            results.append({
                                "concentration": exp["concentration"],
                                "replicate": exp["replicat"],
                                "receptor_area": rec_area,
                                "complex_area": comp_area,
                                "ratio_hill": ratio_hill,
                                "ratio_gabelica": ratio_gabelica,
                            })

                        res_df = pd.DataFrame(results)
                        avg_df = (
                            res_df.groupby("concentration")
                            .agg({
                                "receptor_area": ["mean", "std"],
                                "complex_area": ["mean", "std"],
                                "ratio_hill": ["mean", "std"],
                                "ratio_gabelica": ["mean", "std"],
                            })
                            .reset_index()
                        )

                        # Flatten column names
                        avg_df.columns = [
                            "concentration",
                            "receptor_mean", "receptor_std",
                            "complex_mean", "complex_std",
                            "ratio_hill_mean", "ratio_hill_std",
                            "ratio_gabelica_mean", "ratio_gabelica_std"
                        ]

                        # Store in session state
                        st.session_state.avg_df = avg_df
                        st.session_state.analysis_done = True
                        
                        # Force rerun to show outlier exclusion widget
                        st.rerun()

                # Use data from session state
                avg_df = st.session_state.avg_df

                st.subheader("Résultats moyennés")
                display_df = avg_df[["concentration", "receptor_mean", "complex_mean", "ratio_hill_mean", "ratio_gabelica_mean"]].copy()
                display_df.columns = ["Concentration (µM)", "I(A) moyenne", "I(AB) moyenne", "I(AB)/I(A)", "I(A)/I(AB)"]
                st.dataframe(display_df.round(6))

                # Show warning if outliers are excluded
                if len(removed_conc) > 0:
                    st.warning(f"⚠️ Concentrations exclues du fit : {removed_conc}")

                # ----------------------------------------------------------------
                # Prepare for fitting with filtered data
                # ----------------------------------------------------------------
                keep_mask = ~avg_df["concentration"].isin(removed_conc)
                L0 = avg_df.loc[keep_mask, "concentration"].values
                ratio_hill = avg_df.loc[keep_mask, "ratio_hill_mean"].values
                ratio_hill_err = avg_df.loc[keep_mask, "ratio_hill_std"].values
                ratio_gabelica = avg_df.loc[keep_mask, "ratio_gabelica_mean"].values
                ratio_gabelica_err = avg_df.loc[keep_mask, "ratio_gabelica_std"].values

                st.subheader("Diagnostic")
                st.write(f"Points: {len(L0)} (après exclusion éventuelle)")
                if len(L0) > 1:
                    corr_hill = np.corrcoef(L0, ratio_hill)[0, 1]
                    corr_gab = np.corrcoef(L0, ratio_gabelica)[0, 1] if use_gabelica else np.nan
                    st.write(f"Corrélation Hill (conc vs I(AB)/I(A)): {corr_hill:.3f}")
                    if use_gabelica:
                        st.write(f"Corrélation Gabelica (conc vs I(A)/I(AB)): {corr_gab:.3f}")

                # ----------------------------------------------------------------
                # Fit models
                # ----------------------------------------------------------------
                hill_res = None
                gabelica_res = None

                # Hill fitting
                if use_hill:
                    st.subheader("Ajustement du modèle de Hill")
                    min_points = 2 if fix_n_to_1 else 3
                    if len(L0) < min_points:
                        st.warning(f"Au moins {min_points} concentrations distinctes recommandées pour Hill.")
                    else:
                        hill_res = fit_hill(
                            L0,
                            ratio_hill,
                            ratio_hill_err if use_weights else None,
                            use_weights=use_weights,
                            fix_n_to_1=fix_n_to_1
                        )

                        if hill_res is not None:
                            Kd_h, n_h, Rmax_h = hill_res["popt"]
                            err_Kd_h, err_n_h, err_R_h = hill_res["perr"]

                            # Create results table
                            results_data = {
                                "Paramètre": ["KD (µM)", "Coefficient de Hill (n)", "Facteur de réponse (Rmax)"],
                                "Valeur": [f"{Kd_h:.4f}", f"{n_h:.4f}", f"{Rmax_h:.4f}"],
                                "Erreur": [f"± {err_Kd_h:.4f}", f"± {err_n_h:.4f}", f"± {err_R_h:.4f}"],
                                "Statut": [
                                    "Ajusté",
                                    "Fixé à 1.0" if fix_n_to_1 else "Ajusté",
                                    "Ajusté"
                                ]
                            }
                            results_df = pd.DataFrame(results_data)
                            st.dataframe(results_df)

                            # Quality metrics
                            st.write("**Qualité du fit:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("R²", f"{hill_res['r2']:.4f}")
                            with col2:
                                st.metric("AIC", f"{hill_res['aic']:.2f}")
                            with col3:
                                st.metric("BIC", f"{hill_res['bic']:.2f}")

                            # Interpretation of response factor
                            st.write("**Interprétation du facteur de réponse (Hill):**")
                            if Rmax_h > 1.2:
                                st.info(f"Le complexe AB s'ionise {Rmax_h:.1f}× mieux que la cible libre A")
                            elif Rmax_h < 0.8:
                                st.info(f"La cible libre A s'ionise {1/Rmax_h:.1f}× mieux que le complexe AB")
                            else:
                                st.info("Les facteurs de réponse du complexe et de la cible libre sont similaires")
                        else:
                            st.error("Échec de l'ajustement du modèle de Hill")

                # Gabelica fitting (improved)
                if use_gabelica:
                    st.subheader("Ajustement selon la méthode de Gabelica (version améliorée)")
                    if len(L0) < 2:
                        st.warning("Au moins 2 concentrations distinctes recommandées pour Gabelica.")
                    else:
                        # Remove infinite values for Gabelica fit
                        finite_mask = np.isfinite(ratio_gabelica) & (ratio_gabelica > 0)
                        if not np.any(finite_mask):
                            st.error("Pas de données valides pour la méthode de Gabelica (tous les rapports sont infinis ou nuls)")
                        else:
                            gabelica_res = fit_gabelica_corrected(
                                L0[finite_mask],
                                ratio_gabelica[finite_mask],
                                ratio_gabelica_err[finite_mask] if use_weights else None,
                                use_weights=use_weights
                            )

                            if gabelica_res is not None:
                                Kd_g, R_g = gabelica_res["popt"]
                                err_Kd_g, err_R_g = gabelica_res["perr"]
                                Ka_g = gabelica_res["Ka"]

                                # Create results table
                                results_data_g = {
                                    "Paramètre": ["KD (µM)", "Ka (µM⁻¹)", "Facteur de réponse (R)"],
                                    "Valeur": [f"{Kd_g:.4f}", f"{Ka_g:.6f}", f"{R_g:.4f}"],
                                    "Erreur": [f"± {err_Kd_g:.4f}", f"± {gabelica_res['perr_original'][0]:.6f}", f"± {err_R_g:.4f}"]
                                }
                                results_df_g = pd.DataFrame(results_data_g)
                                st.dataframe(results_df_g)

                                # Quality metrics
                                st.write("**Qualité du fit:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("R²", f"{gabelica_res['r2']:.4f}")
                                with col2:
                                    st.metric("AIC", f"{gabelica_res['aic']:.2f}")

                                # NEW: Show diagnostic information
                                if gabelica_res.get("bounds_warning"):
                                    st.warning("⚠️ Avertissements d'ajustement:")
                                    for warning in gabelica_res["bounds_warning"]:
                                        st.write(f" • {warning}")

                                # NEW: Show initial guesses (for debugging)
                                with st.expander("🔍 Informations de diagnostic (avancé)"):
                                    initial_Ka, initial_R = gabelica_res["initial_guess"]
                                    st.write(f"**Estimation initiale:** Ka = {initial_Ka:.6f} µM⁻¹, R = {initial_R:.4f}")
                                    st.write(f"**Valeurs finales:** Ka = {Ka_g:.6f} µM⁻¹, R = {R_g:.4f}")
                                    st.write("**Limites utilisées:** Ka ∈ [1e-10, 1e10], R ∈ [0.001, 1000]")

                                # Interpretation of response factor
                                st.write("**Interprétation du facteur de réponse (Gabelica):**")
                                if R_g > 1.2:
                                    st.info(f"Le complexe AB répond {R_g:.1f}× mieux que la cible libre A (facteur de réponse)")
                                elif R_g < 0.8:
                                    st.info(f"La cible libre A répond {1/R_g:.1f}× mieux que le complexe AB")
                                else:
                                    st.info("Les facteurs de réponse du complexe et de la cible libre sont similaires")

                                st.info("ℹ️ **Note:** Cette version améliorée utilise des limites élargies pour R (0.001-1000 au lieu de 0.1-10) et une estimation initiale intelligente.")
                            else:
                                st.error("Échec de l'ajustement selon la méthode de Gabelica")

                # ----------------------------------------------------------------
                # Fit plots - CORRECTED TO AVOID ASYMPTOTE PROBLEMS
                # ----------------------------------------------------------------
                if hill_res is not None or gabelica_res is not None:
                    st.subheader("Graphiques des ajustements")

                    # IMPORTANT CORRECTION: Start L_smooth from minimum measured concentration
                    # instead of 0 to avoid asymptote problems
                    L_min = np.min(L0[L0 > 0]) * 0.8 # Start slightly below first measured point
                    L_max = max(np.max(L0) * 1.2, np.max(L0))
                    L_smooth = np.linspace(L_min, L_max, 400)

                    # Hill plot
                    if hill_res is not None:
                        st.write("**Modèle de Hill: I(AB)/I(A) vs Concentration**")
                        fig_hill = go.Figure()

                        # Data points
                        fig_hill.add_trace(
                            go.Scatter(
                                x=L0,
                                y=ratio_hill,
                                mode="markers",
                                name="Données",
                                marker=dict(size=10, color="darkslateblue"),
                                error_y=dict(
                                    type="data",
                                    array=ratio_hill_err,
                                    visible=True
                                ) if ratio_hill_err is not None else None
                            )
                        )

                        # Hill fit
                        Kd_h, n_h, Rmax_h = hill_res["popt"]
                        hill_label = f"Hill (KD={Kd_h:.3g} µM"
                        if fix_n_to_1:
                            hill_label += ", n=1 fixé"
                        else:
                            hill_label += f", n={n_h:.3g}"
                        hill_label += f", Rmax={Rmax_h:.3g})"

                        fig_hill.add_trace(
                            go.Scatter(
                                x=L_smooth,
                                y=hill_res["ypred_full"](L_smooth),
                                mode="lines",
                                name=hill_label,
                                line=dict(width=3, color="red"),
                            )
                        )

                        fig_hill.update_layout(
                            xaxis_title="Concentration ligand (µM)",
                            yaxis_title="Ratio I(AB)/I(A)",
                            width=900,
                            height=500,
                        )
                        st.plotly_chart(fig_hill, use_container_width=True)

                    # Gabelica plot
                    if gabelica_res is not None:
                        st.write("**Méthode de Gabelica: I(A)/I(AB) vs Concentration**")
                        fig_gab = go.Figure()

                        # Data points (only finite values)
                        fig_gab.add_trace(
                            go.Scatter(
                                x=L0[finite_mask],
                                y=ratio_gabelica[finite_mask],
                                mode="markers",
                                name="Données",
                                marker=dict(size=10, color="darkgreen"),
                                error_y=dict(
                                    type="data",
                                    array=ratio_gabelica_err[finite_mask],
                                    visible=True
                                ) if ratio_gabelica_err is not None else None
                            )
                        )

                        # Gabelica fit - use same corrected L_smooth range
                        Kd_g, R_g = gabelica_res["popt"]
                        gabelica_label = f"Gabelica (KD={Kd_g:.3g} µM, R={R_g:.3g})"

                        fig_gab.add_trace(
                            go.Scatter(
                                x=L_smooth,
                                y=gabelica_res["ypred_full"](L_smooth),
                                mode="lines",
                                name=gabelica_label,
                                line=dict(width=3, color="blue", dash="dash"),
                            )
                        )

                        fig_gab.update_layout(
                            xaxis_title="Concentration ligand (µM)",
                            yaxis_title="Ratio I(A)/I(AB)",
                            width=900,
                            height=500,
                        )
                        st.plotly_chart(fig_gab, use_container_width=True)

                    # Export data
                    export_data = {
                        "concentration_uM": L0,
                    }

                    if hill_res is not None:
                        export_data["ratio_hill_observed"] = ratio_hill
                        export_data["ratio_hill_err"] = ratio_hill_err if ratio_hill_err is not None else np.nan
                        export_data["hill_predicted"] = hill_res["ypred_full"](L0)
                        export_data["hill_residuals"] = ratio_hill - hill_res["ypred_full"](L0)

                    if gabelica_res is not None:
                        export_data["ratio_gabelica_observed"] = ratio_gabelica
                        export_data["ratio_gabelica_err"] = ratio_gabelica_err if ratio_gabelica_err is not None else np.nan
                        # For export, extend predictions to all points, but mark invalid ones
                        gab_pred = np.full(len(L0), np.nan)
                        gab_pred[finite_mask] = gabelica_res["ypred_full"](L0[finite_mask])
                        export_data["gabelica_predicted"] = gab_pred
                        gab_resid = np.full(len(L0), np.nan)
                        gab_resid[finite_mask] = ratio_gabelica[finite_mask] - gabelica_res["ypred_full"](L0[finite_mask])
                        export_data["gabelica_residuals"] = gab_resid

                    out_df = pd.DataFrame(export_data)
                    csv_buf = io.StringIO()
                    out_df.to_csv(csv_buf, index=False)
                    st.download_button(
                        "Télécharger les résultats (CSV)",
                        csv_buf.getvalue(),
                        file_name="resultats_fit_comparison_improved.csv",
                        mime="text/csv",
                    )

                    # Summary comparison
                    if hill_res is not None and gabelica_res is not None:
                        st.subheader("Comparaison des méthodes")
                        comparison_data = {
                            "Méthode": ["Hill", "Gabelica (améliorée)"],
                            "KD (µM)": [f"{hill_res['popt'][0]:.4f}", f"{gabelica_res['popt'][0]:.4f}"],
                            "R²": [f"{hill_res['r2']:.4f}", f"{gabelica_res['r2']:.4f}"],
                            "AIC": [f"{hill_res['aic']:.2f}", f"{gabelica_res['aic']:.2f}"],
                            "Paramètres": ["3 (Kd, n, Rmax)" if not fix_n_to_1 else "2 (Kd, Rmax)", "2 (Kd, R)"],
                            "Équation": ["I(AB)/I(A)", "I(A)/I(AB)"]
                        }
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df)

                        # Recommend best method
                        if gabelica_res['aic'] < hill_res['aic']:
                            st.success("🎯 La méthode de Gabelica (améliorée) semble mieux ajuster les données (AIC plus faible)")
                        elif hill_res['aic'] < gabelica_res['aic']:
                            st.success("🎯 Le modèle de Hill semble mieux ajuster les données (AIC plus faible)")
                        else:
                            st.info("ℹ️ Les deux méthodes donnent des ajustements similaires")

    else:
        st.markdown(
            """
            ## Instructions

            - **Type de titrage** : Cochez "Titrage équimolaire" si cible et ligand ont les mêmes concentrations
            - **Méthodes** : Choisissez Hill et/ou Gabelica (Gabelica nécessite un titrage équimolaire)
            - **Fichier CSV** : ligne 1 réplicats, ligne 2 concentrations, puis colonnes m/z et intensité
            - **Plages m/z** : Définissez les plages pour la cible (A) et le complexe (AB)
            - **Analyse** : Cliquez **Lancer l'analyse** pour intégrer, ajuster et visualiser les modèles
            - **🗂️ Exclusion d'outliers** : Après l'analyse, utilisez la sidebar pour exclure des points suspects

            ### Méthodes disponibles

            **Modèle de Hill :** Méthode classique avec coopérativité optionnelle
            - Paramètres : KD, coefficient de Hill (n), facteur de réponse (Rmax)
            - Option pour fixer n=1 (pas de coopérativité)
            - Fitte : I(AB)/I(A) vs concentration

            **Méthode de Gabelica (améliorée) :** Méthode spécialisée pour titrages équimolaires (Gabelica et al., 2003)
            - Paramètres : KD (converti de Ka), facteur de réponse (R)
            - Corrige explicitement les différences de facteurs de réponse entre espèces libres et complexées
            - Nécessite un titrage équimolaire [cible] = [Ligand]
            - **Utilise l'équation :** I(A)/I(AB) = (1 + √(1 + 4×Ka×C₀)) / (2×R×Ka×C₀)
            - Fitte : I(A)/I(AB) vs concentration avec Ka, puis convertit en Kd = 1/Ka

            **Le facteur de réponse indique la différence d'efficacité d'ionisation entre le complexe et la cible libre.**
            """
        )

# -----------------------------------------------------------------------------

# Entrypoint

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
