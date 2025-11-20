import pandas as pd
import streamlit as st
import re
import io
import plotly.graph_objs as go
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

st.image("C:/Users/maxim_eujcrxj/OneDrive/Documents/ULiège/Doctorat/Softwares/MSLabTB/MSTB.png", width=150)

# Langue
lang = st.selectbox("Langue / Language", ["Français", "English"])

def t(fr, en):
    return fr if lang == "Français" else en

st.title(t("Visualiseur de courbes de dissociation (Données MS CID)", "Breakdown Curve Viewer (CID MS Data)"))

# ============================================================================
# FONCTIONS UTILITAIRES POUR GÉRER LES VALEURS NaN/Inf
# ============================================================================

def safe_max(data, default=1.0):
    """Calcule le maximum en ignorant NaN et Inf, avec une valeur par défaut"""
    if not data:
        return default
    # Convertir en array numpy et filtrer les valeurs finies
    clean_data = np.array(data)
    finite_data = clean_data[np.isfinite(clean_data)]
    if len(finite_data) == 0:
        return default
    return float(np.max(finite_data))

def safe_min(data, default=0.0):
    """Calcule le minimum en ignorant NaN et Inf, avec une valeur par défaut"""
    if not data:
        return default
    # Convertir en array numpy et filtrer les valeurs finies
    clean_data = np.array(data)
    finite_data = clean_data[np.isfinite(clean_data)]
    if len(finite_data) == 0:
        return default
    return float(np.min(finite_data))

def validate_axis_limits(min_val, max_val, default_min=0.0, default_max=1.0):
    """Valide et corrige les limites d'axes pour matplotlib"""
    # Vérifier si les valeurs sont finies
    if not np.isfinite(min_val):
        min_val = default_min
    if not np.isfinite(max_val):
        max_val = default_max
    # S'assurer que min < max
    if min_val >= max_val:
        max_val = min_val + 0.1
    return float(min_val), float(max_val)

# ============================================================================
# SIGMOID FITTING FUNCTIONS
# ============================================================================

def hill_equation(x, bottom, top, ec50, hill_slope):
    """Hill equation (4-parameter logistic)"""
    return bottom + (top - bottom) / (1 + (x / ec50) ** hill_slope)

def boltzmann_equation(x, bottom, top, v50, slope):
    """Boltzmann sigmoidal equation"""
    return bottom + (top - bottom) / (1 + np.exp((v50 - x) / slope))

def modified_boltzmann(x, bottom, top, v50, slope):
    """Modified Boltzmann equation with intuitive slope parameter"""
    range_val = top - bottom
    if range_val == 0: # Avoid division by zero
        return np.full_like(x, bottom)
    return bottom + range_val / (1 + np.exp(4 * slope * (v50 - x) / range_val))

def statistical_sigmoid(x, bottom, top, v50, slope):
    """Sigmoid statistique simple (équation de Hill adaptée)"""
    return bottom + (top - bottom) / (1 + np.exp(slope * (v50 - x)))

def fit_sigmoid_curve(voltages, survival_yield, method="Hill"):
    """
    Fit sigmoid curve to breakdown data and extract V50
    """
    if len(voltages) != len(survival_yield):
        return {"success": False, "error": "Voltage and survival yield arrays must have same length"}

    if len(voltages) < 4:
        return {"success": False, "error": "Need at least 4 data points for sigmoid fitting"}

    # Remove any NaN or infinite values
    mask = np.isfinite(voltages) & np.isfinite(survival_yield)
    voltages_clean = np.array(voltages)[mask]
    survival_clean = np.array(survival_yield)[mask]

    if len(voltages_clean) < 4:
        return {"success": False, "error": "Not enough valid data points after cleaning"}

    try:
        # Estimate initial parameters
        y_min = np.min(survival_clean)
        y_max = np.max(survival_clean)
        v_mid = voltages_clean[np.argmin(np.abs(survival_clean - (y_min + y_max) / 2))]

        if method == "Hill":
            # Initial guess: [bottom, top, ec50, hill_slope]
            initial_guess = [y_min, y_max, v_mid, 1.0]
            bounds = ([0, 0, np.min(voltages_clean), 0.1],
                     [1, 1, np.max(voltages_clean), 10])
            popt, pcov = curve_fit(hill_equation, voltages_clean, survival_clean,
                                 p0=initial_guess, bounds=bounds, maxfev=5000)
            v50 = popt[2] # ec50 parameter
            param_names = ['Bottom', 'Top', 'V50', 'Hill Slope']

        elif method == "Boltzmann":
            # Initial guess: [bottom, top, v50, slope]
            slope_guess = (np.max(voltages_clean) - np.min(voltages_clean)) / 10
            initial_guess = [y_min, y_max, v_mid, slope_guess]
            bounds = ([0, 0, np.min(voltages_clean), 0.1],
                     [1, 1, np.max(voltages_clean), 50])
            popt, pcov = curve_fit(boltzmann_equation, voltages_clean, survival_clean,
                                 p0=initial_guess, bounds=bounds, maxfev=5000)
            v50 = popt[2] # v50 parameter
            param_names = ['Bottom', 'Top', 'V50', 'Slope']

        elif method == "Modified Boltzmann":
            # Initial guess: [bottom, top, v50, slope]
            slope_guess = 0.01 # smaller for modified version
            initial_guess = [y_min, y_max, v_mid, slope_guess]
            bounds = ([0, 0, np.min(voltages_clean), -1],
                     [1, 1, np.max(voltages_clean), 1])
            popt, pcov = curve_fit(modified_boltzmann, voltages_clean, survival_clean,
                                 p0=initial_guess, bounds=bounds, maxfev=5000)
            v50 = popt[2] # v50 parameter
            param_names = ['Bottom', 'Top', 'V50', 'Slope']

        elif method == "Statistical Sigmoid":
            # Initial guess: [bottom, top, v50, slope]
            slope_guess = 1.0
            initial_guess = [y_min, y_max, v_mid, slope_guess]
            bounds = ([0, 0, np.min(voltages_clean), 0.1],
                     [1, 1, np.max(voltages_clean), 10])
            popt, pcov = curve_fit(statistical_sigmoid, voltages_clean, survival_clean,
                                 p0=initial_guess, bounds=bounds, maxfev=5000)
            v50 = popt[2] # v50 parameter
            param_names = ['Bottom', 'Top', 'V50', 'Slope']

        else:
            return {"success": False, "error": f"Unknown fitting method: {method}"}

        # Calculate R-squared
        if method == "Hill":
            y_pred = hill_equation(voltages_clean, *popt)
        elif method == "Boltzmann":
            y_pred = boltzmann_equation(voltages_clean, *popt)
        elif method == "Modified Boltzmann":
            y_pred = modified_boltzmann(voltages_clean, *popt)
        elif method == "Statistical Sigmoid":
            y_pred = statistical_sigmoid(voltages_clean, *popt)

        residuals = survival_clean - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((survival_clean - np.mean(survival_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate parameter errors from covariance matrix
        param_errors = np.sqrt(np.diag(pcov))

        # Calculate additional fit statistics
        n = len(voltages_clean)
        p = len(popt) # number of parameters

        # Adjusted R²
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

        # Standard error of the estimate
        mse = ss_res / (n - p) if n > p else 0
        rmse = np.sqrt(mse)

        # AIC and BIC
        if mse > 0:
            aic = n * np.log(mse) + 2 * p
            bic = n * np.log(mse) + p * np.log(n)
        else:
            aic = np.inf
            bic = np.inf

        return {
            "success": True,
            "method": method,
            "v50": v50,
            "r2": r2,
            "adj_r2": adj_r2,
            "rmse": rmse,
            "aic": aic,
            "bic": bic,
            "parameters": popt,
            "param_errors": param_errors,
            "param_names": param_names,
            "voltages_clean": voltages_clean,
            "survival_clean": survival_clean,
            "residuals": residuals,
            "y_pred": y_pred
        }

    except Exception as e:
        return {"success": False, "error": f"Fitting failed: {str(e)}"}

def plot_sigmoid_fit(voltages, survival_yield, fit_result):
    """Add sigmoid fit to existing matplotlib plot"""
    if not fit_result["success"]:
        return None

    # Generate smooth curve for plotting
    v_min, v_max = np.min(voltages), np.max(voltages)
    v_smooth = np.linspace(v_min, v_max, 100)

    method = fit_result["method"]
    popt = fit_result["parameters"]

    if method == "Hill":
        y_smooth = hill_equation(v_smooth, *popt)
    elif method == "Boltzmann":
        y_smooth = boltzmann_equation(v_smooth, *popt)
    elif method == "Modified Boltzmann":
        y_smooth = modified_boltzmann(v_smooth, *popt)
    elif method == "Statistical Sigmoid":
        y_smooth = statistical_sigmoid(v_smooth, *popt)

    return v_smooth, y_smooth

# ============================================================================
# NOUVELLE FONCTION POUR L'EXCLUSION DE POINTS
# ============================================================================

def apply_point_exclusions(voltages, data_arrays, excluded_indices):
    """
    Applique les exclusions de points aux données

    Args:
        voltages: liste des voltages
        data_arrays: liste des arrays de données (survival_yield, fragments)
        excluded_indices: set des indices à exclure

    Returns:
        voltages_filtered, data_arrays_filtered
    """
    if not excluded_indices:
        return voltages, data_arrays

    # Créer un masque pour garder les points non-exclus
    mask = np.ones(len(voltages), dtype=bool)
    for idx in excluded_indices:
        if 0 <= idx < len(voltages):
            mask[idx] = False

    # Appliquer le masque
    voltages_filtered = [v for i, v in enumerate(voltages) if mask[i]]
    data_arrays_filtered = []

    for data_array in data_arrays:
        filtered_array = [d for i, d in enumerate(data_array) if mask[i]]
        data_arrays_filtered.append(filtered_array)

    return voltages_filtered, data_arrays_filtered

# Upload CSV file
uploaded_file = st.file_uploader(t("Téléverse ton fichier CSV", "Upload your CSV file"), type=["csv"])

# Encoding selection
encoding_option = st.selectbox(t("Choisis l'encodage du fichier", "Select file encoding"), options=["utf-8", "latin1"], index=0)

if uploaded_file:
    # Sélection du format de données
    data_format = st.selectbox(
        t("Format des données CSV", "CSV data format"),
        ["TWIMExtract", "Manuel"],
        index=0,
        help=t(
            "TWIMExtract: format standard avec '$TrapCV:' en première colonne\nManuel: colonnes alternées M/Z et Intensité pour chaque voltage",
            "TWIMExtract: standard format with '$TrapCV:' in first column\nManuel: alternating M/Z and Intensity columns for each voltage"
        )
    )

    try:
        if data_format == "TWIMExtract":
            # Traitement format TWIMExtract (code original)
            df = pd.read_csv(uploaded_file, header=2, encoding=encoding_option)

            # Vérification format fichier
            if str(df.columns[0]).strip() != "$TrapCV:":
                st.error(t("Format invalide : première colonne doit être '$TrapCV:'", "Invalid format: first column must be '$TrapCV:'"))
                st.stop()

            # Extraction m/z (float) avec gestion des NaN
            mz_values = pd.to_numeric(df.iloc[:, 0], errors='coerce').values

            # Extraction voltages (convertit 4.0 -> 4, 6.0 -> 6, etc.)
            voltage_labels = df.columns[1:]
            voltages = []

            for label in voltage_labels:
                try:
                    voltage = int(float(str(label).strip()))
                    voltages.append(voltage)
                except Exception:
                    st.warning(t(f"Colonne ignorée : {label}", f"Ignored column: {label}"))

            # Construction voltage_dict
            voltage_dict = {}
            for idx, volt in enumerate(voltages):
                intensity_values = pd.to_numeric(df.iloc[:, idx + 1], errors='coerce').values
                voltage_dict[volt] = {
                    'mz': mz_values,
                    'intensity': intensity_values
                }

        else: # Format Manuel
            # Traitement format Manuel (nouveau format)
            df = pd.read_csv(uploaded_file, encoding=encoding_option)

            # Identification des colonnes : alternance M/Z, Intensité, M/Z, Intensité, etc.
            voltage_dict = {}
            voltages = []

            # Parcourir les colonnes par paires
            for col_idx in range(1, len(df.columns), 2): # Colonnes d'intensité (indices impairs : 1, 3, 5, ...)
                if col_idx >= len(df.columns):
                    break

                mz_col_idx = col_idx - 1 # Colonne M/Z correspondante (indices pairs : 0, 2, 4, ...)
                intensity_col_idx = col_idx # Colonne intensité

                # Extraire le voltage depuis l'en-tête de la colonne d'intensité
                intensity_col_name = str(df.columns[intensity_col_idx]).strip()

                try:
                    # Le voltage devrait être directement la valeur numérique dans l'en-tête
                    voltage = int(float(intensity_col_name))
                    voltages.append(voltage)

                    # Extraire les données M/Z et intensité avec gestion des NaN
                    mz_values = pd.to_numeric(df.iloc[:, mz_col_idx], errors='coerce').values
                    intensity_values = pd.to_numeric(df.iloc[:, intensity_col_idx], errors='coerce').values

                    voltage_dict[voltage] = {
                        'mz': mz_values,
                        'intensity': intensity_values
                    }

                except Exception as e:
                    st.warning(t(f"Erreur lors du traitement des colonnes {mz_col_idx+1}-{intensity_col_idx+1} (voltage: {intensity_col_name}) : {e}",
                                f"Error processing columns {mz_col_idx+1}-{intensity_col_idx+1} (voltage: {intensity_col_name}): {e}"))

    except Exception as e:
        st.error(t(f"Erreur de lecture CSV : {e}", f"CSV read error: {e}"))
        st.stop()

    if not voltage_dict:
        st.error(t("Aucun voltage valide détecté", "No valid voltages found"))
        st.stop()

    # Tri des voltages
    sorted_voltages = sorted(voltages)

    # Sélection des voltages à afficher
    all_voltages_option = st.checkbox(t("Afficher toutes les courbes", "Show all curves"))
    voltages_to_plot = sorted_voltages if all_voltages_option else [
        st.selectbox(t("Choisis un voltage", "Select voltage"), sorted_voltages)
    ]

    # Spectre MS
    st.subheader(t("Spectre MS", "MS Spectrum"))

    # Calcul zoom automatique avec gestion des NaN
    all_mz = []
    for volt in voltages_to_plot:
        mz_data = voltage_dict[volt]['mz']
        # Filtrer les valeurs finies
        finite_mz = mz_data[np.isfinite(mz_data)]
        all_mz.extend(finite_mz)

    mz_min_auto = safe_min(all_mz, 0)
    mz_max_auto = safe_max(all_mz, 1000)

    # Personnalisation axe m/z
    st.markdown(t("**Personnalisation de l'axe m/z pour le spectre MS**", "**m/z axis customization for MS Spectrum**"))
    zoom_range = st.text_input(t("Intervalle m/z (min,max)", "m/z range (min,max)"), f"{mz_min_auto:.1f},{mz_max_auto:.1f}")

    try:
        mz_min, mz_max = map(float, zoom_range.split(","))
        # Valider les limites
        mz_min, mz_max = validate_axis_limits(mz_min, mz_max, mz_min_auto, mz_max_auto)
    except:
        mz_min, mz_max = mz_min_auto, mz_max_auto

    # Graphique Plotly
    fig = go.Figure()

    for volt in voltages_to_plot:
        data = voltage_dict[volt]
        # Filtrer les données pour Plotly (enlever NaN)
        mask = np.isfinite(data['mz']) & np.isfinite(data['intensity'])
        fig.add_trace(go.Scatter(
            x=data['mz'][mask],
            y=data['intensity'][mask],
            mode='lines',
            name=f"{volt} V"
        ))

    fig.update_layout(
        title=t("Spectre MS", "MS Spectrum"),
        xaxis_title="m/z",
        yaxis_title="Intensité",
        xaxis_range=[mz_min, mz_max]
    )

    st.plotly_chart(fig, use_container_width=True)

    # Courbes de dissociation
    st.subheader(t("Courbes de dissociation", "Breakdown Curves"))

    # Sélection plages
    precursor_min = st.number_input(t("m/z précurseur min", "Precursor min m/z"), value=float(mz_min))
    precursor_max = st.number_input(t("m/z précurseur max", "Precursor max m/z"), value=float(mz_max))

    if precursor_min >= precursor_max:
        st.error(t("Erreur : min >= max", "Error: min >= max"))
        st.stop()

    # Fragments
    num_fragments = st.number_input(t("Nombre de fragments", "Number of fragments"), 0, 5, 2)
    fragment_ranges = []
    fragment_labels = []

    for i in range(num_fragments):
        col1, col2 = st.columns(2)
        with col1:
            min_val = st.number_input(f"Fragment {i+1} min", value=100.0 + i*10)
        with col2:
            max_val = st.number_input(f"Fragment {i+1} max", value=110.0 + i*10)

        frag_label = st.text_input(f"Légende du Fragment {i+1}", value=f"Fragment {i+1}")
        fragment_ranges.append((min_val, max_val))
        fragment_labels.append(frag_label)

    # Calcul des intensités avec gestion robuste des NaN
    survival_yield = []
    fragment_intensities = [[] for _ in range(num_fragments)]

    for volt in sorted_voltages:
        data = voltage_dict[volt]

        # Filtrer les données valides (mz et intensity finies)
        mask = np.isfinite(data['mz']) & np.isfinite(data['intensity'])
        valid_mz = data['mz'][mask]
        valid_intensity = data['intensity'][mask]

        # Calculer l'intensité totale
        total_intensity = np.sum(valid_intensity) if len(valid_intensity) > 0 else 0

        # Survival yield
        if total_intensity > 0:
            precursor_int = np.sum([
                i for mz, i in zip(valid_mz, valid_intensity)
                if precursor_min <= mz <= precursor_max
            ])
            survival_yield.append(precursor_int / total_intensity)
        else:
            survival_yield.append(0.0)

        # Fragments — normalisation comme le parent
        for idx, (frag_min, frag_max) in enumerate(fragment_ranges):
            if total_intensity > 0:
                frag_int = np.sum([
                    i for mz, i in zip(valid_mz, valid_intensity)
                    if frag_min <= mz <= frag_max
                ])
                fragment_intensities[idx].append(frag_int / total_intensity)
            else:
                fragment_intensities[idx].append(0.0)

    # ============================================================================
    # NOUVELLE SECTION: EXCLUSION DE POINTS
    # ============================================================================

    st.subheader(t("🚫 Exclusion de points", "🚫 Point Exclusion"))

    enable_exclusion = st.checkbox(t("✅ Activer l'exclusion de points", "✅ Enable point exclusion"))
    excluded_indices = set()

    if enable_exclusion:
        st.markdown(t("**Sélection des points à exclure de l'analyse et de l'ajustement:**", "**Select points to exclude from analysis and fitting:**"))

        # Afficher un tableau avec les données pour faciliter la sélection
        data_preview = pd.DataFrame({
            t("Index", "Index"): range(len(sorted_voltages)),
            t("Voltage", "Voltage"): sorted_voltages,
            t("Survival Yield", "Survival Yield"): [f"{sy:.4f}" for sy in survival_yield]
        })

        # Ajouter les colonnes de fragments s'il y en a
        for idx, label in enumerate(fragment_labels):
            data_preview[label] = [f"{fi:.4f}" for fi in fragment_intensities[idx]]

        st.dataframe(data_preview, use_container_width=True)

        # Méthode de sélection des points à exclure
        exclusion_method = st.radio(
            t("Méthode de sélection", "Selection method"),
            [t("Liste d'indices", "Index list"), t("Sélection par voltage", "Voltage selection")]
        )

        if exclusion_method == t("Liste d'indices", "Index list"):
            indices_input = st.text_input(
                t("Indices à exclure (séparés par des virgules)", "Indices to exclude (comma-separated)"),
                help=t("Exemple: 0,2,5 pour exclure les points aux indices 0, 2 et 5", 
                      "Example: 0,2,5 to exclude points at indices 0, 2 and 5")
            )

            if indices_input.strip():
                try:
                    indices = [int(x.strip()) for x in indices_input.split(',') if x.strip()]
                    # Valider les indices
                    valid_indices = [i for i in indices if 0 <= i < len(sorted_voltages)]
                    invalid_indices = [i for i in indices if i not in valid_indices]

                    if invalid_indices:
                        st.warning(t(f"Indices invalides ignorés: {invalid_indices}", 
                                   f"Invalid indices ignored: {invalid_indices}"))

                    excluded_indices = set(valid_indices)

                except ValueError:
                    st.error(t("Format d'indices invalide", "Invalid index format"))

        else:  # Sélection par voltage
            voltages_to_exclude = st.multiselect(
                t("Voltages à exclure", "Voltages to exclude"),
                options=sorted_voltages,
                help=t("Sélectionnez les voltages dont vous voulez exclure les points", 
                      "Select the voltages whose points you want to exclude")
            )

            excluded_indices = set([sorted_voltages.index(v) for v in voltages_to_exclude])

        # Affichage du résumé des exclusions
        if excluded_indices:
            excluded_voltages = [sorted_voltages[i] for i in excluded_indices]
            st.info(t(f"Points exclus: indices {sorted(excluded_indices)} (voltages: {excluded_voltages})", 
                     f"Excluded points: indices {sorted(excluded_indices)} (voltages: {excluded_voltages})"))

            # Vérifier qu'il reste assez de points pour l'analyse
            remaining_points = len(sorted_voltages) - len(excluded_indices)
            if remaining_points < 3:
                st.error(t(f"Erreur: seulement {remaining_points} points restants. Au moins 3 points sont nécessaires pour l'analyse.",
                          f"Error: only {remaining_points} points remaining. At least 3 points are needed for analysis."))
                excluded_indices = set()  # Réinitialiser les exclusions

    # ============================================================================
    # APPLICATION DES EXCLUSIONS AUX DONNÉES
    # ============================================================================

    # Appliquer les exclusions si activées
    if enable_exclusion and excluded_indices:
        all_data_arrays = [survival_yield] + fragment_intensities
        voltages_filtered, data_arrays_filtered = apply_point_exclusions(
            sorted_voltages, all_data_arrays, excluded_indices
        )

        survival_yield_filtered = data_arrays_filtered[0]
        fragment_intensities_filtered = data_arrays_filtered[1:]

        st.success(t(f"✅ {len(excluded_indices)} point(s) exclus. {len(voltages_filtered)} points restants pour l'analyse.",
                    f"✅ {len(excluded_indices)} point(s) excluded. {len(voltages_filtered)} points remaining for analysis."))
    else:
        # Utiliser toutes les données
        voltages_filtered = sorted_voltages.copy()
        survival_yield_filtered = survival_yield.copy()
        fragment_intensities_filtered = [fi.copy() for fi in fragment_intensities]

    # ============================================================================
    # SIGMOID FITTING SECTION AVEC SIGMOID STATISTIQUE SIMPLE
    # ============================================================================

    st.subheader(t("🔬 Ajustement sigmoïde et extraction V50", "🔬 Sigmoid Fitting and V50 Extraction"))

    # Options de fit sigmoïde
    enable_sigmoid = st.checkbox(t("✅ Activer l'ajustement sigmoïde", "✅ Enable sigmoid fitting"))

    if enable_sigmoid:
        col1, col2 = st.columns(2)

        with col1:
            sigmoid_method = st.selectbox(
                t("Méthode d'ajustement", "Fitting method"),
                ["Hill", "Boltzmann", "Modified Boltzmann", "Statistical Sigmoid"],
                help=t("Hill: équation de Hill classique\nBoltzmann: équation de Boltzmann\nModified Boltzmann: Boltzmann modifié\nStatistical Sigmoid: sigmoid statistique simple",
                      "Hill: classical Hill equation\nBoltzmann: Boltzmann equation\nModified Boltzmann: modified Boltzmann\nStatistical Sigmoid: simple statistical sigmoid")
            )

        with col2:
            fit_target = st.selectbox(
                t("Courbe à ajuster", "Curve to fit"),
                [t("Survival Yield", "Survival Yield")] + fragment_labels
            )

        # Déterminer les données à ajuster (utilisez les données filtrées)
        if fit_target == t("Survival Yield", "Survival Yield"):
            data_to_fit = survival_yield_filtered
        else:
            frag_idx = fragment_labels.index(fit_target)
            data_to_fit = fragment_intensities_filtered[frag_idx]

        # Effectuer le fit
        if len(voltages_filtered) >= 4 and len(data_to_fit) >= 4:
            fit_result = fit_sigmoid_curve(voltages_filtered, data_to_fit, sigmoid_method)

            if fit_result["success"]:
                # Afficher les résultats du fit
                st.success(t("🎯 Ajustement réussi !", "🎯 Fitting successful!"))

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        t("V50", "V50"),
                        f"{fit_result['v50']:.2f} V",
                        help=t("Voltage de demi-fragmentation", "Half-fragmentation voltage")
                    )

                with col2:
                    st.metric("R²", f"{fit_result['r2']:.4f}",
                             help=t("Coefficient de détermination", "Coefficient of determination"))

                with col3:
                    st.metric(t("R² ajusté", "Adj. R²"), f"{fit_result['adj_r2']:.4f}",
                             help=t("R² ajusté pour le nombre de paramètres", "R² adjusted for number of parameters"))

                with col4:
                    st.metric("RMSE", f"{fit_result['rmse']:.4f}",
                             help=t("Erreur quadratique moyenne", "Root mean square error"))

                # Tableau des paramètres
                st.subheader(t("📊 Paramètres de l'ajustement", "📊 Fitting Parameters"))

                param_df = pd.DataFrame({
                    t("Paramètre", "Parameter"): fit_result['param_names'],
                    t("Valeur", "Value"): [f"{p:.4f}" for p in fit_result['parameters']],
                    t("Erreur Standard", "Standard Error"): [f"±{e:.4f}" for e in fit_result['param_errors']]
                })

                st.dataframe(param_df, use_container_width=True)

                # Critères d'information
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("AIC", f"{fit_result['aic']:.2f}",
                             help=t("Critère d'information d'Akaike (plus bas = meilleur)", "Akaike Information Criterion (lower = better)"))

                with col2:
                    st.metric("BIC", f"{fit_result['bic']:.2f}",
                             help=t("Critère d'information bayésien (plus bas = meilleur)", "Bayesian Information Criterion (lower = better)"))

                with col3:
                    st.metric(t("Méthode", "Method"), fit_result['method'])

            else:
                st.error(t(f"❌ Échec de l'ajustement : {fit_result['error']}",
                          f"❌ Fitting failed: {fit_result['error']}"))
                fit_result = None

        else:
            st.warning(t("⚠️ Besoin d'au moins 4 points de données pour l'ajustement sigmoïde",
                        "⚠️ Need at least 4 data points for sigmoid fitting"))
            fit_result = None
    else:
        fit_result = None

    # Personnalisation des axes breakdown curve AVEC PROTECTION CONTRE NaN/Inf
    st.markdown(t("**Personnalisation des axes pour la courbe de dissociation**", "**Breakdown curve axis customization**"))

    # Calculs sécurisés des limites par défaut (utiliser les données filtrées pour un meilleur zoom)
    default_y_min = safe_min(survival_yield_filtered, 0.0)
    default_y_max = safe_max(survival_yield_filtered, 1.0)
    default_x_min = min(voltages_filtered) if voltages_filtered else 0
    default_x_max = max(voltages_filtered) if voltages_filtered else 50

    breakdown_y_min = st.number_input(
        t("Intensité min (axe Y)", "Intensity min (Y axis)"),
        value=float(default_y_min)
    )

    breakdown_y_max = st.number_input(
        t("Intensité max (axe Y)", "Intensity max (Y axis)"),
        value=float(default_y_max)
    )

    x_step = st.number_input(t("Pas de voltage (axe X)", "Voltage step (X axis)"), min_value=1, value=2)
    x_min = st.number_input(t("Voltage min (axe X)", "Min voltage (X axis)"), value=int(default_x_min))
    x_max = st.number_input(t("Voltage max (axe X)", "Max voltage (X axis)"), value=int(default_x_max))

    # VALIDATION FINALE DES LIMITES AVANT LE GRAPHIQUE
    breakdown_y_min, breakdown_y_max = validate_axis_limits(breakdown_y_min, breakdown_y_max, 0.0, 1.0)
    x_min, x_max = validate_axis_limits(x_min, x_max, default_x_min, default_x_max)

    # Graphique Matplotlib AVEC PROTECTION ET FIT SIGMOÏDE
    fig2, ax = plt.subplots()

    # Afficher TOUTES les données (y compris les points exclus) mais les distinguer visuellement
    # Points inclus dans l'analyse
    valid_voltages = [v for v, s in zip(voltages_filtered, survival_yield_filtered) if np.isfinite(s)]
    valid_survival = [s for s in survival_yield_filtered if np.isfinite(s)]

    if len(valid_voltages) > 0 and len(valid_survival) > 0:
        ax.plot(valid_voltages, valid_survival, 'o-', label=t("Survival Yield (inclus)", "Survival Yield (included)"), 
                linewidth=2, markersize=6)

    # Points exclus (si il y en a)
    if enable_exclusion and excluded_indices:
        excluded_voltages_plot = [sorted_voltages[i] for i in excluded_indices]
        excluded_survival_plot = [survival_yield[i] for i in excluded_indices if np.isfinite(survival_yield[i])]

        if len(excluded_voltages_plot) > 0 and len(excluded_survival_plot) > 0:
            ax.plot(excluded_voltages_plot, excluded_survival_plot, 'x', 
                   label=t("Survival Yield (exclus)", "Survival Yield (excluded)"), 
                   color='red', markersize=8, markeredgewidth=2)

    # Fragments (données filtrées pour l'analyse)
    for idx in range(num_fragments):
        valid_frag_voltages = [v for v, f in zip(voltages_filtered, fragment_intensities_filtered[idx]) if np.isfinite(f)]
        valid_frag_intensities = [f for f in fragment_intensities_filtered[idx] if np.isfinite(f)]

        if len(valid_frag_voltages) > 0 and len(valid_frag_intensities) > 0:
            ax.plot(valid_frag_voltages, valid_frag_intensities, 'x--', 
                   label=f"{fragment_labels[idx]} (inclus)", alpha=0.8)

        # Fragments exclus
        if enable_exclusion and excluded_indices:
            excluded_frag_voltages = [sorted_voltages[i] for i in excluded_indices]
            excluded_frag_intensities = [fragment_intensities[idx][i] for i in excluded_indices if np.isfinite(fragment_intensities[idx][i])]

            if len(excluded_frag_voltages) > 0 and len(excluded_frag_intensities) > 0:
                ax.plot(excluded_frag_voltages, excluded_frag_intensities, 's', 
                       label=f"{fragment_labels[idx]} (exclus)", 
                       color='red', markersize=6, alpha=0.6)

    # Ajouter le fit sigmoïde si activé (utilise les données filtrées)
    if enable_sigmoid and fit_result and fit_result["success"]:
        smooth_data = plot_sigmoid_fit(voltages_filtered, data_to_fit, fit_result)
        if smooth_data:
            v_smooth, y_smooth = smooth_data
            ax.plot(v_smooth, y_smooth, 'r-', linewidth=3,
                   label=f"📈 {fit_result['method']} (V50={fit_result['v50']:.1f}V, R²={fit_result['r2']:.3f})", 
                   alpha=0.8)

    ax.set_xlabel(t("Voltage de collision (V)", "Collision voltage (V)"))
    ax.set_ylabel(t("Survival Yield", "Survival yield"))

    # Application sécurisée des limites
    try:
        ax.set_ylim([breakdown_y_min, breakdown_y_max])
        ax.set_xlim([x_min, x_max])
        ax.set_xticks(list(range(int(x_min), int(x_max)+1, int(x_step))))
    except Exception as e:
        st.error(t(f"Erreur lors de la configuration des axes : {e}", f"Error configuring axes: {e}"))
        # Limites par défaut en cas d'erreur
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 50])

    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # Export SVG de la breakdown curve
    st.subheader(t("Export SVG de la courbe de dissociation", "Export breakdown curve as SVG"))

    if st.button(t("Exporter la courbe en SVG", "Export breakdown curve as SVG")):
        svg_buffer = io.BytesIO()
        fig2.savefig(svg_buffer, format='svg')
        svg_buffer.seek(0)

        st.download_button(
            label=t("Télécharger SVG", "Download SVG"),
            data=svg_buffer.getvalue(),
            file_name="breakdown_curve_with_fit.svg",
            mime="image/svg+xml"
        )

    # Export Excel avec résultats du fit
    if st.button(t("Exporter Excel", "Export Excel")):
        # Nettoyer les données avant l'export (utiliser les données originales)
        clean_data = {
            "Voltage": sorted_voltages,
            "Survival_Yield": [s if np.isfinite(s) else 0.0 for s in survival_yield],
            "Excluded": [i in excluded_indices for i in range(len(sorted_voltages))]
        }

        for idx in range(num_fragments):
            clean_data[fragment_labels[idx]] = [
                f if np.isfinite(f) else 0.0 for f in fragment_intensities[idx]
            ]

        df_export = pd.DataFrame(clean_data)

        # Ajouter les résultats du fit si disponibles
        if enable_sigmoid and fit_result and fit_result["success"]:
            # Créer le fichier Excel avec plusieurs feuilles
            excel_buffer = io.BytesIO()

            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_export.to_excel(writer, sheet_name='Breakdown_Data', index=False)

                # Ajouter une feuille avec les paramètres du fit
                df_fit_params = pd.DataFrame({
                    t("Paramètre", "Parameter"): fit_result['param_names'],
                    t("Valeur", "Value"): fit_result['parameters'],
                    t("Erreur", "Error"): fit_result['param_errors']
                })
                df_fit_params.to_excel(writer, sheet_name='Sigmoid_Fit_Parameters', index=False)

                # Ajouter les résultats de fit dans une feuille séparée
                df_results = pd.DataFrame({
                    'Metric': ['V50', 'R²', 'Adj_R²', 'RMSE', 'AIC', 'BIC', 'Method', 'Points_Used', 'Points_Excluded'],
                    'Value': [fit_result['v50'], fit_result['r2'], fit_result['adj_r2'],
                             fit_result['rmse'], fit_result['aic'], fit_result['bic'], fit_result['method'],
                             len(voltages_filtered), len(excluded_indices)]
                })
                df_results.to_excel(writer, sheet_name='Fit_Results', index=False)

        else:
            excel_buffer = io.BytesIO()
            df_export.to_excel(excel_buffer, index=False)

        st.download_button(
            label=t("Télécharger Excel", "Download Excel"),
            data=excel_buffer.getvalue(),
            file_name="data_with_sigmoid_fit.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.markdown("---")
st.markdown(t("Développé avec <3 par M.P.V. Benonit. Modifié avec ajustement sigmoïde statistique et exclusion de points.",
             "Developed with <3 by M.P.V. Benonit. Modified with statistical sigmoid fitting and point exclusion."))
