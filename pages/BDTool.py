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

def app():
    st.image("images/MSTB.png", width=150)
    
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
        if data is None:
            return default
        try:
            if len(data) == 0:
                return default
        except TypeError:
            return default
        # Convertir en array numpy et filtrer les valeurs finies
        clean_data = np.array(data)
        finite_data = clean_data[np.isfinite(clean_data)]
        if len(finite_data) == 0:
            return default
        return float(np.max(finite_data))
    
    def safe_min(data, default=0.0):
        """Calcule le minimum en ignorant NaN et Inf, avec une valeur par défaut"""
        if data is None:
            return default
        try:
            if len(data) == 0:
                return default
        except TypeError:
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
            # Scale-aware fitting: keep bounded fit stable even if data is normalized to 100%.
            y_scale = float(np.max(np.abs(survival_clean)))
            if not np.isfinite(y_scale) or y_scale <= 0:
                return {"success": False, "error": "Invalid signal amplitude for fitting"}

            fit_scaled = y_scale > (1.0 + 1e-9)
            survival_for_fit = survival_clean / y_scale if fit_scaled else survival_clean.copy()

            # Estimate initial parameters
            y_min = np.min(survival_for_fit)
            y_max = np.max(survival_for_fit)
            v_mid = voltages_clean[np.argmin(np.abs(survival_for_fit - (y_min + y_max) / 2))]
    
            if method == "Hill":
                # Initial guess: [bottom, top, ec50, hill_slope]
                initial_guess = [y_min, y_max, v_mid, 1.0]
                bounds = ([0, 0, np.min(voltages_clean), 0.1],
                         [1, 1, np.max(voltages_clean), 10])
                popt, pcov = curve_fit(hill_equation, voltages_clean, survival_for_fit,
                                     p0=initial_guess, bounds=bounds, maxfev=5000)
                v50 = popt[2] # ec50 parameter
                param_names = ['Bottom', 'Top', 'V50', 'Hill Slope']
    
            elif method == "Boltzmann":
                # Initial guess: [bottom, top, v50, slope]
                slope_guess = (np.max(voltages_clean) - np.min(voltages_clean)) / 10
                initial_guess = [y_min, y_max, v_mid, slope_guess]
                bounds = ([0, 0, np.min(voltages_clean), 0.1],
                         [1, 1, np.max(voltages_clean), 50])
                popt, pcov = curve_fit(boltzmann_equation, voltages_clean, survival_for_fit,
                                     p0=initial_guess, bounds=bounds, maxfev=5000)
                v50 = popt[2] # v50 parameter
                param_names = ['Bottom', 'Top', 'V50', 'Slope']
    
            elif method == "Statistical Sigmoid":
                # Initial guess: [bottom, top, v50, slope]
                slope_guess = 1.0
                initial_guess = [y_min, y_max, v_mid, slope_guess]
                bounds = ([0, 0, np.min(voltages_clean), 0.1],
                         [1, 1, np.max(voltages_clean), 10])
                popt, pcov = curve_fit(statistical_sigmoid, voltages_clean, survival_for_fit,
                                     p0=initial_guess, bounds=bounds, maxfev=5000)
                v50 = popt[2] # v50 parameter
                param_names = ['Bottom', 'Top', 'V50', 'Slope']
    
            else:
                return {"success": False, "error": f"Unknown fitting method: {method}"}
    
            # Calculate R-squared
            if method == "Hill":
                y_pred_fit = hill_equation(voltages_clean, *popt)
            elif method == "Boltzmann":
                y_pred_fit = boltzmann_equation(voltages_clean, *popt)
            elif method == "Statistical Sigmoid":
                y_pred_fit = statistical_sigmoid(voltages_clean, *popt)

            y_pred = y_pred_fit * y_scale if fit_scaled else y_pred_fit
    
            residuals = survival_clean - y_pred
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((survival_clean - np.mean(survival_clean)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
            # Calculate parameter errors from covariance matrix
            param_errors = np.sqrt(np.diag(pcov))

            # Report bottom/top parameters on the original y scale.
            popt_report = popt.copy()
            if fit_scaled:
                popt_report[0] = popt_report[0] * y_scale
                popt_report[1] = popt_report[1] * y_scale
                param_errors[0] = param_errors[0] * y_scale
                param_errors[1] = param_errors[1] * y_scale
    
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
                "parameters": popt_report,
                "param_errors": param_errors,
                "param_names": param_names,
                "voltages_clean": voltages_clean,
                "survival_clean": survival_clean,
                "residuals": residuals,
                "y_pred": y_pred,
                "y_scale": y_scale
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
        elif method == "Statistical Sigmoid":
            y_smooth = statistical_sigmoid(v_smooth, *popt)
    
        return v_smooth, y_smooth

    def evaluate_sigmoid_curve(x_values, fit_result):
        """Evaluate fitted sigmoid model for arbitrary x values."""
        if not fit_result or (not fit_result.get("success", False)):
            return None

        method = fit_result.get("method")
        params = fit_result.get("parameters", [])
        if params is None or len(params) < 4:
            return None

        x_arr = np.asarray(x_values, dtype=float)
        try:
            if method == "Hill":
                return hill_equation(x_arr, *params)
            if method == "Boltzmann":
                return boltzmann_equation(x_arr, *params)
            if method == "Statistical Sigmoid":
                return statistical_sigmoid(x_arr, *params)
        except Exception:
            return None

        return None

    def compute_sigmoid_derivative(x_values, fit_result):
        """Compute analytical dy/dV for the fitted sigmoid model."""
        if not fit_result or (not fit_result.get("success", False)):
            return None

        method = fit_result.get("method")
        params = fit_result.get("parameters", [])
        if params is None or len(params) < 4:
            return None

        x_arr = np.asarray(x_values, dtype=float)
        bottom, top, v50, shape = [float(p) for p in params[:4]]

        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            if method == "Hill":
                ec50 = v50 if abs(v50) > 1e-12 else (1e-12 if v50 >= 0 else -1e-12)
                base = np.clip(x_arr / ec50, 1e-12, None)
                base_h = np.power(base, shape)
                base_hm1 = np.power(base, shape - 1.0)
                deriv = -((top - bottom) * shape * base_hm1) / (ec50 * np.power(1.0 + base_h, 2.0))
                return deriv

            if method == "Boltzmann":
                slope = shape if abs(shape) > 1e-12 else (1e-12 if shape >= 0 else -1e-12)
                z = np.clip((v50 - x_arr) / slope, -700, 700)
                ez = np.exp(z)
                deriv = ((top - bottom) * ez) / (slope * np.power(1.0 + ez, 2.0))
                return deriv

            if method == "Statistical Sigmoid":
                k = shape
                z = np.clip(k * (v50 - x_arr), -700, 700)
                ez = np.exp(z)
                deriv = ((top - bottom) * k * ez) / np.power(1.0 + ez, 2.0)
                return deriv

        return None

    def compute_transition_metrics(fit_result, x_min=None, x_max=None):
        """Compute steep-transition metrics from a fitted sigmoid model."""
        if not fit_result or (not fit_result.get("success", False)):
            return {"success": False, "error": "No successful fit available"}

        voltages_clean = np.asarray(fit_result.get("voltages_clean", []), dtype=float)
        if len(voltages_clean) < 4:
            return {"success": False, "error": "Not enough fitted points"}

        if x_min is None:
            x_min = safe_min(voltages_clean, 0.0)
        if x_max is None:
            x_max = safe_max(voltages_clean, x_min + 1.0)
        if (not np.isfinite(x_min)) or (not np.isfinite(x_max)) or (x_min >= x_max):
            x_min = safe_min(voltages_clean, 0.0)
            x_max = safe_max(voltages_clean, x_min + 1.0)
            if x_min >= x_max:
                x_max = x_min + 1.0

        x_dense = np.linspace(float(x_min), float(x_max), 2000)
        y_dense = evaluate_sigmoid_curve(x_dense, fit_result)
        dy_dense = compute_sigmoid_derivative(x_dense, fit_result)

        if y_dense is None or dy_dense is None:
            return {"success": False, "error": "Unable to evaluate fitted model"}

        y_dense = np.asarray(y_dense, dtype=float)
        dy_dense = np.asarray(dy_dense, dtype=float)

        dy_at_v50_arr = compute_sigmoid_derivative(np.asarray([fit_result.get("v50", np.nan)], dtype=float), fit_result)
        slope_at_v50 = float(dy_at_v50_arr[0]) if (dy_at_v50_arr is not None and len(dy_at_v50_arr) > 0) else np.nan

        steepest_slope = np.nan
        steepest_slope_abs = np.nan
        steepest_voltage = np.nan
        finite_deriv = np.isfinite(dy_dense)
        if np.any(finite_deriv):
            abs_dy = np.abs(dy_dense)
            abs_dy[~finite_deriv] = -np.inf
            idx_max = int(np.argmax(abs_dy))
            steepest_slope = float(dy_dense[idx_max])
            steepest_slope_abs = float(abs_dy[idx_max])
            steepest_voltage = float(x_dense[idx_max])

        params = np.asarray(fit_result.get("parameters", []), dtype=float)
        v10 = np.nan
        v90 = np.nan
        v10_v90_width = np.nan
        if len(params) >= 2:
            bottom = float(params[0])
            top = float(params[1])
            amplitude = top - bottom
            if np.isfinite(amplitude) and abs(amplitude) > 1e-12:
                progress = (y_dense - bottom) / amplitude
                finite_progress = np.isfinite(progress) & np.isfinite(x_dense)
                if np.count_nonzero(finite_progress) >= 2:
                    p = progress[finite_progress]
                    xv = x_dense[finite_progress]

                    if p[-1] >= p[0]:
                        p_mono = np.maximum.accumulate(p)
                        p_interp = p_mono
                        x_interp = xv
                    else:
                        p_mono = np.minimum.accumulate(p)
                        p_interp = p_mono[::-1]
                        x_interp = xv[::-1]

                    p_unique, unique_idx = np.unique(p_interp, return_index=True)
                    x_unique = x_interp[unique_idx]
                    if len(p_unique) >= 2 and p_unique[0] <= 0.1 <= p_unique[-1] and p_unique[0] <= 0.9 <= p_unique[-1]:
                        v10 = float(np.interp(0.1, p_unique, x_unique))
                        v90 = float(np.interp(0.9, p_unique, x_unique))
                        v10_v90_width = float(abs(v90 - v10))

        return {
            "success": True,
            "slope_at_v50": slope_at_v50,
            "steepest_slope": steepest_slope,
            "steepest_slope_abs": steepest_slope_abs,
            "steepest_voltage": steepest_voltage,
            "v10": v10,
            "v90": v90,
            "v10_v90_width": v10_v90_width,
        }
    
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

    def normalize_series(values, target_max=1.0):
        """Normalise une serie finie sur son maximum (ignore NaN/Inf)."""
        arr = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(arr)

        if not np.any(finite_mask):
            return arr.tolist(), 1.0

        ref_max = float(np.max(arr[finite_mask]))
        out = np.full(arr.shape, np.nan, dtype=float)

        if ref_max > 0:
            out[finite_mask] = (arr[finite_mask] / ref_max) * target_max
        else:
            out[finite_mask] = 0.0
            ref_max = 1.0

        return out.tolist(), ref_max

    def normalize_with_reference(values, reference_max, target_max=1.0):
        """Applique une normalisation avec une reference donnee (pour inclure/exclure de facon coherente)."""
        arr = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(arr)
        out = np.full(arr.shape, np.nan, dtype=float)

        if reference_max > 0:
            out[finite_mask] = (arr[finite_mask] / reference_max) * target_max
        else:
            out[finite_mask] = 0.0

        return out.tolist()

    def compute_breakdown_arrays(voltage_dict, sorted_voltages, precursor_min, precursor_max, fragment_ranges):
        """Calcule survival yield et fragments pour un jeu de donnees."""
        survival_vals = []
        fragment_vals = [[] for _ in range(len(fragment_ranges))]

        for volt in sorted_voltages:
            data = voltage_dict[volt]

            # Filtrer les données valides (mz et intensity finies)
            mask = np.isfinite(data['mz']) & np.isfinite(data['intensity'])
            valid_mz = data['mz'][mask]
            valid_intensity = data['intensity'][mask]

            total_intensity = np.sum(valid_intensity) if len(valid_intensity) > 0 else 0

            # Survival yield
            if total_intensity > 0:
                precursor_int = np.sum([
                    i for mz, i in zip(valid_mz, valid_intensity)
                    if precursor_min <= mz <= precursor_max
                ])
                survival_vals.append(precursor_int / total_intensity)
            else:
                survival_vals.append(0.0)

            # Fragments
            for idx, (frag_min, frag_max) in enumerate(fragment_ranges):
                if total_intensity > 0:
                    frag_int = np.sum([
                        i for mz, i in zip(valid_mz, valid_intensity)
                        if frag_min <= mz <= frag_max
                    ])
                    fragment_vals[idx].append(frag_int / total_intensity)
                else:
                    fragment_vals[idx].append(0.0)

        return survival_vals, fragment_vals
    
    # Upload CSV file(s)
    uploaded_files = st.file_uploader(
        t("Téléverse un ou plusieurs fichiers CSV", "Upload one or more CSV files"),
        type=["csv"],
        accept_multiple_files=True
    )
    
    # Encoding selection
    encoding_option = st.selectbox(t("Choisis l'encodage du fichier", "Select file encoding"), options=["UTF-8", "latin1"], index=0)
    
    if uploaded_files:
        # Sélection du format de données
        data_format = st.selectbox(
            t("Format des données CSV", "CSV data format"),
            ["TWIMExtract", "Manuel", "FelionyX Batch Extract"],
            index=0,
            help=t(
                "TWIMExtract: format standard avec '$TrapCV:' en première colonne\nManuel: colonnes alternees M/Z et Intensite pour chaque voltage\nFelionyX Batch Extract: 1ere ligne = voltages (ex: 20V), 2e ligne = m/z et Intensite par paires",
                "TWIMExtract: standard format with '$TrapCV:' in first column\nManuel: alternating M/Z and Intensity columns for each voltage\nFelionyX Batch Extract: 1st row = voltages (e.g. 20V), 2nd row = m/z and Intensity in pairs"
            )
        )

        def normalize_voltage_value(value):
            value = float(value)
            return int(value) if float(value).is_integer() else value

        def parse_voltage_from_label(label):
            label_str = str(label).strip()
            if not label_str:
                return None

            # Accept labels like "36", "36.0", "36V", "36.0 V".
            match = re.search(r'(-?\d+(?:[\.,]\d+)?)\s*[Vv]?$', label_str)
            if not match:
                return None

            try:
                return normalize_voltage_value(match.group(1).replace(',', '.'))
            except Exception:
                return None

        def detect_separator(sample_text):
            sep_counts = {
                ';': sample_text.count(';'),
                ',': sample_text.count(','),
                '\t': sample_text.count('\t')
            }
            best_sep = max(sep_counts, key=sep_counts.get)
            return best_sep if sep_counts[best_sep] > 0 else ','

        def parse_one_dataset(file_obj):
            try:
                file_obj.seek(0)
                sample_text = file_obj.read(16384).decode(encoding_option, errors='replace')
                file_obj.seek(0)
                detected_sep = detect_separator(sample_text)
            except Exception as e:
                raise ValueError(t(f"Erreur de lecture du fichier : {e}", f"File read error: {e}"))

            def read_csv_flexible(header='infer'):
                try:
                    file_obj.seek(0)
                    return pd.read_csv(file_obj, header=header, sep=detected_sep, encoding=encoding_option)
                except Exception:
                    file_obj.seek(0)
                    # Fallback for unusual CSVs where separator is inconsistent.
                    return pd.read_csv(file_obj, header=header, sep=None, engine='python', encoding=encoding_option)

            voltage_dict = {}
            voltages = []

            if data_format == "TWIMExtract":
                df = read_csv_flexible(header=2)
                if str(df.columns[0]).strip() != "$TrapCV:":
                    raise ValueError(t("Format invalide : première colonne doit être '$TrapCV:'", "Invalid format: first column must be '$TrapCV:'"))

                mz_values = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
                voltage_labels = df.columns[1:]

                for label in voltage_labels:
                    try:
                        voltage = int(float(str(label).strip()))
                        voltages.append(voltage)
                    except Exception:
                        st.warning(f"[{file_obj.name}] " + t(f"Colonne ignorée : {label}", f"Ignored column: {label}"))

                for idx, volt in enumerate(voltages):
                    intensity_values = pd.to_numeric(df.iloc[:, idx + 1], errors='coerce').values
                    voltage_dict[volt] = {
                        'mz': mz_values,
                        'intensity': intensity_values
                    }

            elif data_format == "Manuel":
                df = read_csv_flexible()

                for col_idx in range(1, len(df.columns), 2):
                    if col_idx >= len(df.columns):
                        break

                    mz_col_idx = col_idx - 1
                    intensity_col_idx = col_idx
                    intensity_col_name = str(df.columns[intensity_col_idx]).strip()

                    try:
                        voltage = int(float(intensity_col_name))
                        voltages.append(voltage)

                        mz_values = pd.to_numeric(df.iloc[:, mz_col_idx], errors='coerce').values
                        intensity_values = pd.to_numeric(df.iloc[:, intensity_col_idx], errors='coerce').values

                        voltage_dict[voltage] = {
                            'mz': mz_values,
                            'intensity': intensity_values
                        }
                    except Exception as e:
                        st.warning(
                            f"[{file_obj.name}] " +
                            t(f"Erreur lors du traitement des colonnes {mz_col_idx+1}-{intensity_col_idx+1} (voltage: {intensity_col_name}) : {e}",
                              f"Error processing columns {mz_col_idx+1}-{intensity_col_idx+1} (voltage: {intensity_col_name}): {e}")
                        )

            else:
                df_raw = read_csv_flexible(header=None)

                if df_raw.shape[0] < 3 or df_raw.shape[1] < 2:
                    raise ValueError(
                        t(
                            "Format FelionyX invalide : fichier trop court (au moins 3 lignes et 2 colonnes).",
                            "Invalid FelionyX format: file too short (at least 3 rows and 2 columns)."
                        )
                    )

                title_row = df_raw.iloc[0]
                header_row = df_raw.iloc[1].astype(str).str.strip().str.lower()
                data_df = df_raw.iloc[2:].reset_index(drop=True)

                for mz_col_idx in range(0, data_df.shape[1], 2):
                    intensity_col_idx = mz_col_idx + 1
                    if intensity_col_idx >= data_df.shape[1]:
                        continue

                    voltage_label = title_row.iloc[mz_col_idx]
                    voltage = parse_voltage_from_label(voltage_label)
                    if voltage is None:
                        voltage = parse_voltage_from_label(title_row.iloc[intensity_col_idx])

                    if voltage is None:
                        st.warning(
                            f"[{file_obj.name}] " +
                            t(
                                f"Paire de colonnes ignorée ({mz_col_idx+1}-{intensity_col_idx+1}) : voltage non détecté ({voltage_label}).",
                                f"Ignored column pair ({mz_col_idx+1}-{intensity_col_idx+1}): no voltage detected ({voltage_label})."
                            )
                        )
                        continue

                    mz_header = header_row.iloc[mz_col_idx] if mz_col_idx < len(header_row) else ""
                    int_header = header_row.iloc[intensity_col_idx] if intensity_col_idx < len(header_row) else ""
                    if ("m/z" not in mz_header and "mz" not in mz_header) or ("int" not in int_header):
                        st.warning(
                            f"[{file_obj.name}] " +
                            t(
                                f"Paire {voltage}V importée malgré en-têtes inattendus: '{mz_header}' / '{int_header}'.",
                                f"Pair {voltage}V imported despite unexpected headers: '{mz_header}' / '{int_header}'."
                            )
                        )

                    mz_values = pd.to_numeric(data_df.iloc[:, mz_col_idx], errors='coerce').values
                    intensity_values = pd.to_numeric(data_df.iloc[:, intensity_col_idx], errors='coerce').values

                    voltage_dict[voltage] = {
                        'mz': mz_values,
                        'intensity': intensity_values
                    }
                    voltages.append(voltage)

            if not voltage_dict:
                raise ValueError(t("Aucun voltage valide détecté", "No valid voltages found"))

            return {
                'name': file_obj.name,
                'voltage_dict': voltage_dict,
                'sorted_voltages': sorted(set(voltages))
            }

        datasets = []
        for file_obj in uploaded_files:
            try:
                datasets.append(parse_one_dataset(file_obj))
            except Exception as e:
                st.warning(f"[{file_obj.name}] {e}")

        if not datasets:
            st.error(t("Aucun fichier valide détecté", "No valid dataset could be loaded"))
            st.stop()

        dataset_names = [d['name'] for d in datasets]

        if len(datasets) > 1:
            st.info(
                t(
                    f"{len(datasets)} jeux de données chargés. Sélectionnez le jeu principal pour l'analyse détaillée puis utilisez la section comparaison.",
                    f"{len(datasets)} datasets loaded. Select the primary dataset for detailed analysis, then use the comparison section."
                )
            )
            selected_dataset_name = st.selectbox(
                t("Jeu de données principal", "Primary dataset"),
                dataset_names,
                index=0
            )
        else:
            selected_dataset_name = dataset_names[0]

        active_dataset = next(d for d in datasets if d['name'] == selected_dataset_name)
        voltage_dict = active_dataset['voltage_dict']
        sorted_voltages = active_dataset['sorted_voltages']

        st.caption(t(f"Jeu actif: {selected_dataset_name}", f"Active dataset: {selected_dataset_name}"))
    
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
        survival_yield, fragment_intensities = compute_breakdown_arrays(
            voltage_dict,
            sorted_voltages,
            precursor_min,
            precursor_max,
            fragment_ranges
        )
    
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
        # NORMALISATION OPTIONNELLE DES COURBES
        # ============================================================================

        st.subheader(t("⚖️ Normalisation des intensités", "⚖️ Intensity normalization"))
        enable_normalization = st.checkbox(
            t("✅ Normaliser l'intensité de la courbe de dissociation", "✅ Normalize breakdown-curve intensity"),
            value=False
        )

        norm_label_max1 = t("Maximum = 1.0", "Maximum = 1.0")
        norm_label_max100 = t("Maximum = 100%", "Maximum = 100%")
        selected_norm_scale = norm_label_max1

        if enable_normalization:
            selected_norm_scale = st.radio(
                t("Échelle de normalisation", "Normalization scale"),
                [norm_label_max1, norm_label_max100],
                horizontal=True
            )

        normalization_target_max = 100.0 if (enable_normalization and selected_norm_scale == norm_label_max100) else 1.0

        if enable_normalization:
            survival_yield_plot, survival_ref_max = normalize_series(survival_yield_filtered, normalization_target_max)

            fragment_intensities_plot = []
            fragment_ref_maxes = []
            for frag_values in fragment_intensities_filtered:
                frag_norm, frag_ref = normalize_series(frag_values, normalization_target_max)
                fragment_intensities_plot.append(frag_norm)
                fragment_ref_maxes.append(frag_ref)

            # Normaliser aussi les donnees completes pour afficher correctement les points exclus.
            survival_yield_all_plot = normalize_with_reference(survival_yield, survival_ref_max, normalization_target_max)
            fragment_intensities_all_plot = []
            for idx, frag_values_all in enumerate(fragment_intensities):
                ref = fragment_ref_maxes[idx] if idx < len(fragment_ref_maxes) else 1.0
                fragment_intensities_all_plot.append(normalize_with_reference(frag_values_all, ref, normalization_target_max))

            st.info(
                t(
                    f"Intensites normalisees sur un maximum de {normalization_target_max:g} par courbe.",
                    f"Intensities normalized to a maximum of {normalization_target_max:g} per curve."
                )
            )
        else:
            survival_yield_plot = survival_yield_filtered.copy()
            fragment_intensities_plot = [fi.copy() for fi in fragment_intensities_filtered]
            survival_yield_all_plot = survival_yield.copy()
            fragment_intensities_all_plot = [fi.copy() for fi in fragment_intensities]

        # ============================================================================
        # COMPARAISON MULTI-DATASETS
        # ============================================================================

        if len(datasets) > 1:
            st.subheader(t("📊 Comparaison entre jeux de données", "📊 Multi-dataset comparison"))

            compare_curve_options = [t("Survival Yield", "Survival Yield")] + fragment_labels
            compare_curve_target = st.selectbox(
                t("Courbe à comparer", "Curve to compare"),
                compare_curve_options,
                key="compare_curve_target"
            )

            compare_dataset_names = st.multiselect(
                t("Jeux de données à comparer", "Datasets to compare"),
                options=dataset_names,
                default=dataset_names,
                key="compare_dataset_names"
            )

            if not compare_dataset_names:
                st.warning(t("Sélectionnez au moins un jeu de données.", "Select at least one dataset."))
            else:
                comparison_payload = []
                compare_rows = []

                for ds in datasets:
                    if ds['name'] not in compare_dataset_names:
                        continue

                    ds_survival, ds_fragments = compute_breakdown_arrays(
                        ds['voltage_dict'],
                        ds['sorted_voltages'],
                        precursor_min,
                        precursor_max,
                        fragment_ranges
                    )

                    if compare_curve_target == t("Survival Yield", "Survival Yield"):
                        y_values = ds_survival
                    else:
                        frag_idx_cmp = fragment_labels.index(compare_curve_target)
                        y_values = ds_fragments[frag_idx_cmp]

                    if enable_normalization:
                        y_values, _ = normalize_series(y_values, normalization_target_max)

                    valid_pairs = [
                        (x, y)
                        for x, y in zip(ds['sorted_voltages'], y_values)
                        if np.isfinite(y)
                    ]

                    if not valid_pairs:
                        continue

                    xv = [p[0] for p in valid_pairs]
                    yv = [p[1] for p in valid_pairs]
                    comparison_payload.append({
                        "name": ds['name'],
                        "x": xv,
                        "y": yv
                    })
                    for x_val, y_val in valid_pairs:
                        compare_rows.append({
                            "Dataset": ds['name'],
                            "Voltage": x_val,
                            "Value": y_val
                        })

                if comparison_payload:
                    all_cmp_x = [x for row in comparison_payload for x in row["x"]]
                    all_cmp_y = [y for row in comparison_payload for y in row["y"]]

                    cmp_default_y_limit = normalization_target_max if enable_normalization else 1.0
                    cmp_default_y_min = safe_min(all_cmp_y, 0.0)
                    cmp_default_y_max = safe_max(all_cmp_y, cmp_default_y_limit)
                    cmp_default_x_min = safe_min(all_cmp_x, 0.0)
                    cmp_default_x_max = safe_max(all_cmp_x, 50.0)

                    st.markdown(t(
                        "**Personnalisation des axes pour le graphe de comparaison**",
                        "**Axis customization for comparison graph**"
                    ))

                    cmp_col1, cmp_col2, cmp_col3 = st.columns(3)
                    with cmp_col1:
                        cmp_y_min = st.number_input(
                            t("Intensité min (comparaison)", "Intensity min (comparison)"),
                            value=float(cmp_default_y_min),
                            key="cmp_y_min"
                        )
                    with cmp_col2:
                        cmp_y_max = st.number_input(
                            t("Intensité max (comparaison)", "Intensity max (comparison)"),
                            value=float(cmp_default_y_max),
                            key="cmp_y_max"
                        )
                    with cmp_col3:
                        cmp_x_step = st.number_input(
                            t("Pas de voltage (comparaison)", "Voltage step (comparison)"),
                            min_value=0.1,
                            value=2.0,
                            step=0.1,
                            key="cmp_x_step"
                        )

                    cmp_col4, cmp_col5 = st.columns(2)
                    with cmp_col4:
                        cmp_x_min = st.number_input(
                            t("Voltage min (comparaison)", "Min voltage (comparison)"),
                            value=float(cmp_default_x_min),
                            key="cmp_x_min"
                        )
                    with cmp_col5:
                        cmp_x_max = st.number_input(
                            t("Voltage max (comparaison)", "Max voltage (comparison)"),
                            value=float(cmp_default_x_max),
                            key="cmp_x_max"
                        )

                    cmp_y_min, cmp_y_max = validate_axis_limits(cmp_y_min, cmp_y_max, 0.0, cmp_default_y_limit)
                    cmp_x_min, cmp_x_max = validate_axis_limits(cmp_x_min, cmp_x_max, cmp_default_x_min, cmp_default_x_max)
                    if cmp_x_step <= 0:
                        cmp_x_step = 1.0

                    cmp_graph_title = st.text_input(t("Titre du graphique (comparaison)", "Graph title (comparison)"), value="", key="cmp_graph_title")
                    cmp_x_axis_label = st.text_input(t("Nom de l'axe X (comparaison)", "X axis label (comparison)"), value=t("Voltage de collision (V)", "Collision voltage (V)"), key="cmp_x_axis_label")
                    
                    if enable_normalization:
                        cmp_default_y_label = t("Intensité normalisée (%)", "Normalized intensity (%)") if normalization_target_max == 100.0 else t("Intensité normalisée (max=1)", "Normalized intensity (max=1)")
                    else:
                        cmp_default_y_label = t("Ratio d'intensité", "Intensity ratio")
                    cmp_y_axis_label = st.text_input(t("Nom de l'axe Y (comparaison)", "Y axis label (comparison)"), value=cmp_default_y_label, key="cmp_y_axis_label")
                    
                    cmp_show_legend = st.checkbox(t("Afficher la légende", "Show legend"), value=True, key="cmp_show_legend")

                    st.markdown(t("**Noms des courbes (comparaison)**", "**Curve names (comparison)**"))
                    custom_curve_names = {}
                    for row in comparison_payload:
                        new_name = st.text_input(f"{t('Nom pour', 'Name for')} '{row['name']}'", value=row['name'], key=f"name_{row['name']}")
                        custom_curve_names[row['name']] = new_name

                    enable_compare_sigmoid = st.checkbox(
                        t("✅ Afficher l'ajustement sigmoïde sur le graphe de comparaison", "✅ Show sigmoid fit on comparison graph"),
                        value=False,
                        key="enable_compare_sigmoid"
                    )
                    compare_sigmoid_method = "Hill"
                    enable_compare_transition_metrics = False
                    selected_compare_transition_keys = []

                    compare_transition_metric_map = {
                        t("Pente à V50", "Slope at V50"): "slope_at_v50",
                        t("Pente maximale (zone raide)", "Steepest slope"): "steepest_slope",
                        t("Largeur V10-V90", "V10-V90 width"): "v10_v90_width",
                    }

                    if enable_compare_sigmoid:
                        compare_sigmoid_method = st.selectbox(
                            t("Méthode d'ajustement (comparaison)", "Fitting method (comparison)"),
                            ["Hill", "Boltzmann", "Statistical Sigmoid"],
                            key="compare_sigmoid_method"
                        )
                        enable_compare_transition_metrics = st.checkbox(
                            t("✅ Afficher les métriques de pente en comparaison", "✅ Show slope metrics in comparison"),
                            value=False,
                            key="enable_compare_transition_metrics"
                        )
                        if enable_compare_transition_metrics:
                            selected_compare_labels = st.multiselect(
                                t("Métriques à afficher (comparaison)", "Metrics to display (comparison)"),
                                options=list(compare_transition_metric_map.keys()),
                                default=list(compare_transition_metric_map.keys()),
                                key="compare_transition_metrics_selected"
                            )
                            selected_compare_transition_keys = [
                                compare_transition_metric_map[label]
                                for label in selected_compare_labels
                            ]

                    fig_cmp, ax_cmp = plt.subplots()
                    fit_summary_rows = []

                    for row in comparison_payload:
                        display_name = custom_curve_names.get(row["name"], row["name"])
                        line_obj, = ax_cmp.plot(row["x"], row["y"], 'o-', linewidth=2, markersize=5, label=display_name)

                        if enable_compare_sigmoid:
                            if len(row["x"]) >= 4:
                                cmp_fit = fit_sigmoid_curve(row["x"], row["y"], compare_sigmoid_method)
                                if cmp_fit["success"]:
                                    smooth_data = plot_sigmoid_fit(row["x"], row["y"], cmp_fit)
                                    if smooth_data:
                                        v_smooth_cmp, y_smooth_cmp = smooth_data
                                        ax_cmp.plot(
                                            v_smooth_cmp,
                                            y_smooth_cmp,
                                            '--',
                                            linewidth=1.8,
                                            color=line_obj.get_color(),
                                            alpha=0.9,
                                            label=f"{display_name} fit (V50={cmp_fit['v50']:.1f}V)"
                                        )

                                    fit_row = {
                                        "Dataset": display_name,
                                        "Method": compare_sigmoid_method,
                                        "Status": "OK",
                                        "V50": cmp_fit["v50"],
                                        "R2": cmp_fit["r2"],
                                        "RMSE": cmp_fit["rmse"]
                                    }

                                    if selected_compare_transition_keys:
                                        tr_metrics = compute_transition_metrics(cmp_fit)
                                        if tr_metrics.get("success", False):
                                            if "slope_at_v50" in selected_compare_transition_keys:
                                                fit_row["Slope_at_V50"] = tr_metrics.get("slope_at_v50", np.nan)
                                            if "steepest_slope" in selected_compare_transition_keys:
                                                fit_row["Steepest_Slope"] = tr_metrics.get("steepest_slope_abs", np.nan)
                                            if "v10_v90_width" in selected_compare_transition_keys:
                                                fit_row["V10_V90_Width"] = tr_metrics.get("v10_v90_width", np.nan)
                                        else:
                                            if "slope_at_v50" in selected_compare_transition_keys:
                                                fit_row["Slope_at_V50"] = np.nan
                                            if "steepest_slope" in selected_compare_transition_keys:
                                                fit_row["Steepest_Slope"] = np.nan
                                            if "v10_v90_width" in selected_compare_transition_keys:
                                                fit_row["V10_V90_Width"] = np.nan

                                    fit_summary_rows.append(fit_row)
                                else:
                                    failed_row = {
                                        "Dataset": display_name,
                                        "Method": compare_sigmoid_method,
                                        "Status": "Failed",
                                        "V50": np.nan,
                                        "R2": np.nan,
                                        "RMSE": np.nan,
                                        "Error": cmp_fit.get("error", "Unknown error")
                                    }
                                    if "slope_at_v50" in selected_compare_transition_keys:
                                        failed_row["Slope_at_V50"] = np.nan
                                    if "steepest_slope" in selected_compare_transition_keys:
                                        failed_row["Steepest_Slope"] = np.nan
                                    if "v10_v90_width" in selected_compare_transition_keys:
                                        failed_row["V10_V90_Width"] = np.nan
                                    fit_summary_rows.append(failed_row)
                            else:
                                insufficient_row = {
                                    "Dataset": display_name,
                                    "Method": compare_sigmoid_method,
                                    "Status": "Insufficient points",
                                    "V50": np.nan,
                                    "R2": np.nan,
                                    "RMSE": np.nan
                                }
                                if "slope_at_v50" in selected_compare_transition_keys:
                                    insufficient_row["Slope_at_V50"] = np.nan
                                if "steepest_slope" in selected_compare_transition_keys:
                                    insufficient_row["Steepest_Slope"] = np.nan
                                if "v10_v90_width" in selected_compare_transition_keys:
                                    insufficient_row["V10_V90_Width"] = np.nan
                                fit_summary_rows.append(insufficient_row)

                    if cmp_graph_title.strip():
                        ax_cmp.set_title(cmp_graph_title)
                    else:
                        ax_cmp.set_title(t("Comparaison des courbes de dissociation", "Breakdown curve comparison"))

                    ax_cmp.set_xlabel(cmp_x_axis_label)
                    ax_cmp.set_ylabel(cmp_y_axis_label)

                    try:
                        ax_cmp.set_ylim([cmp_y_min, cmp_y_max])
                        ax_cmp.set_xlim([cmp_x_min, cmp_x_max])
                        cmp_ticks = np.arange(cmp_x_min, cmp_x_max + (cmp_x_step * 0.5), cmp_x_step)
                        if len(cmp_ticks) <= 400:
                            ax_cmp.set_xticks(cmp_ticks)
                    except Exception as e:
                        st.warning(t(
                            f"Configuration automatique des axes de comparaison utilisée ({e}).",
                            f"Automatic comparison-axis configuration used ({e})."
                        ))

                    ax_cmp.grid(True, alpha=0.3)
                    if cmp_show_legend:
                        ax_cmp.legend()
                    st.pyplot(fig_cmp)
                    st.caption(t(
                        "La comparaison utilise les points complets de chaque jeu (sans exclusion de points).",
                        "Comparison uses full points for each dataset (without point exclusion)."
                    ))

                    if enable_compare_sigmoid and fit_summary_rows:
                        st.markdown(t("**Résumé des ajustements (comparaison)**", "**Fit summary (comparison)**"))
                        fit_summary_df = pd.DataFrame(fit_summary_rows)
                        st.dataframe(fit_summary_df, use_container_width=True)
                    else:
                        fit_summary_df = None

                    if compare_rows:
                        compare_df = pd.DataFrame(compare_rows)
                        compare_df["Curve"] = compare_curve_target
                        compare_df["Normalization_Enabled"] = enable_normalization
                        compare_df["Normalization_Max"] = normalization_target_max if enable_normalization else np.nan

                        compare_wide_df = compare_df.pivot_table(
                            index="Voltage",
                            columns="Dataset",
                            values="Value",
                            aggfunc="first"
                        ).reset_index()

                        metadata_df = pd.DataFrame([
                            {"Parameter": "Curve", "Value": compare_curve_target},
                            {"Parameter": "Normalization_Enabled", "Value": enable_normalization},
                            {"Parameter": "Normalization_Max", "Value": normalization_target_max if enable_normalization else "None"},
                            {"Parameter": "Compare_X_Min", "Value": cmp_x_min},
                            {"Parameter": "Compare_X_Max", "Value": cmp_x_max},
                            {"Parameter": "Compare_X_Step", "Value": cmp_x_step},
                            {"Parameter": "Compare_Y_Min", "Value": cmp_y_min},
                            {"Parameter": "Compare_Y_Max", "Value": cmp_y_max},
                            {"Parameter": "Sigmoid_Enabled", "Value": enable_compare_sigmoid},
                            {"Parameter": "Sigmoid_Method", "Value": compare_sigmoid_method if enable_compare_sigmoid else "None"},
                            {"Parameter": "Transition_Metrics_Enabled", "Value": enable_compare_transition_metrics if enable_compare_sigmoid else False},
                            {"Parameter": "Transition_Metrics_Selected", "Value": ",".join(selected_compare_transition_keys) if selected_compare_transition_keys else "None"},
                        ])

                        svg_cmp_buffer = io.BytesIO()
                        fig_cmp.savefig(svg_cmp_buffer, format='svg')
                        svg_cmp_buffer.seek(0)

                        excel_cmp_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_cmp_buffer, engine='xlsxwriter') as writer:
                            compare_df.to_excel(writer, sheet_name='Comparison_Long', index=False)
                            compare_wide_df.to_excel(writer, sheet_name='Comparison_Wide', index=False)
                            metadata_df.to_excel(writer, sheet_name='Comparison_Metadata', index=False)
                            if fit_summary_df is not None:
                                fit_summary_df.to_excel(writer, sheet_name='Comparison_Fits', index=False)
                        excel_cmp_buffer.seek(0)

                        col_cmp_export_1, col_cmp_export_2 = st.columns(2)
                        with col_cmp_export_1:
                            st.download_button(
                                label=t("Télécharger comparaison SVG", "Download comparison SVG"),
                                data=svg_cmp_buffer.getvalue(),
                                file_name="comparison_breakdown_curves.svg",
                                mime="image/svg+xml",
                                key="download_compare_svg"
                            )
                        with col_cmp_export_2:
                            st.download_button(
                                label=t("Télécharger comparaison Excel", "Download comparison Excel"),
                                data=excel_cmp_buffer.getvalue(),
                                file_name="comparison_breakdown_curves.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_compare_excel"
                            )
                else:
                    st.warning(t("Aucune donnée valide à afficher pour la comparaison.", "No valid data to display for comparison."))
    
        # ============================================================================
        # SIGMOID FITTING SECTION AVEC SIGMOID STATISTIQUE SIMPLE
        # ============================================================================
    
        st.subheader(t("🔬 Ajustement sigmoïde et extraction V50", "🔬 Sigmoid Fitting and V50 Extraction"))
    
        # Options de fit sigmoïde
        enable_sigmoid = st.checkbox(t("✅ Activer l'ajustement sigmoïde", "✅ Enable sigmoid fitting"))

        transition_metrics_result = None
        selected_transition_metric_keys = []
    
        if enable_sigmoid:
            col1, col2 = st.columns(2)
    
            with col1:
                sigmoid_method = st.selectbox(
                    t("Méthode d'ajustement", "Fitting method"),
                    ["Hill", "Boltzmann", "Statistical Sigmoid"],
                    help=t("Hill: équation de Hill classique\nBoltzmann: équation de Boltzmann\nStatistical Sigmoid: sigmoid statistique simple",
                          "Hill: classical Hill equation\nBoltzmann: Boltzmann equation\nStatistical Sigmoid: simple statistical sigmoid")
                )
    
            with col2:
                fit_target = st.selectbox(
                    t("Courbe à ajuster", "Curve to fit"),
                    [t("Survival Yield", "Survival Yield")] + fragment_labels
                )
    
            # Déterminer les données à ajuster (utilisez les données filtrées)
            if fit_target == t("Survival Yield", "Survival Yield"):
                data_to_fit = survival_yield_plot
            else:
                frag_idx = fragment_labels.index(fit_target)
                data_to_fit = fragment_intensities_plot[frag_idx]
    
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

                    transition_metric_map = {
                        t("Pente à V50", "Slope at V50"): "slope_at_v50",
                        t("Pente maximale (zone raide)", "Steepest slope"): "steepest_slope",
                        t("Largeur V10-V90", "V10-V90 width"): "v10_v90_width",
                    }

                    enable_transition_metrics = st.checkbox(
                        t("✅ Afficher les métriques de pente/transition", "✅ Show slope/transition metrics"),
                        value=False,
                        key="enable_transition_metrics_main"
                    )

                    if enable_transition_metrics:
                        selected_transition_labels = st.multiselect(
                            t("Métriques à afficher", "Metrics to display"),
                            options=list(transition_metric_map.keys()),
                            default=list(transition_metric_map.keys()),
                            key="selected_transition_metrics_main"
                        )
                        selected_transition_metric_keys = [
                            transition_metric_map[label]
                            for label in selected_transition_labels
                        ]

                        if selected_transition_metric_keys:
                            transition_metrics_result = compute_transition_metrics(fit_result)
                            if transition_metrics_result.get("success", False):
                                transition_rows = []

                                if "slope_at_v50" in selected_transition_metric_keys:
                                    transition_rows.append({
                                        t("Métrique", "Metric"): t("Pente à V50", "Slope at V50"),
                                        t("Valeur", "Value"): transition_metrics_result.get("slope_at_v50", np.nan),
                                        t("Détail", "Detail"): t("Unités: intensité/V", "Units: intensity/V")
                                    })

                                if "steepest_slope" in selected_transition_metric_keys:
                                    transition_rows.append({
                                        t("Métrique", "Metric"): t("Pente maximale (zone raide)", "Steepest slope"),
                                        t("Valeur", "Value"): transition_metrics_result.get("steepest_slope_abs", np.nan),
                                        t("Détail", "Detail"): t(
                                            f"à V={transition_metrics_result.get('steepest_voltage', np.nan):.2f} (intensité/V)",
                                            f"at V={transition_metrics_result.get('steepest_voltage', np.nan):.2f} (intensity/V)"
                                        )
                                    })

                                if "v10_v90_width" in selected_transition_metric_keys:
                                    transition_rows.append({
                                        t("Métrique", "Metric"): t("Largeur V10-V90", "V10-V90 width"),
                                        t("Valeur", "Value"): transition_metrics_result.get("v10_v90_width", np.nan),
                                        t("Détail", "Detail"): t(
                                            f"V10={transition_metrics_result.get('v10', np.nan):.2f}, V90={transition_metrics_result.get('v90', np.nan):.2f}",
                                            f"V10={transition_metrics_result.get('v10', np.nan):.2f}, V90={transition_metrics_result.get('v90', np.nan):.2f}"
                                        )
                                    })

                                if transition_rows:
                                    st.markdown(t("**📐 Métriques de transition**", "**📐 Transition metrics**"))
                                    st.dataframe(pd.DataFrame(transition_rows), use_container_width=True, hide_index=True)
                            else:
                                st.warning(t(
                                    f"Métriques de transition indisponibles: {transition_metrics_result.get('error', 'erreur inconnue')}",
                                    f"Transition metrics unavailable: {transition_metrics_result.get('error', 'unknown error')}"
                                ))
    
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
        st.markdown(t("**Personnalisation des axes et du graphique pour la courbe de dissociation**", "**Breakdown curve axis and graph customization**"))
        
        # Titre et labels
        graph_title = st.text_input(t("Titre du graphique", "Graph title"), value="")
        x_axis_label = st.text_input(t("Nom de l'axe X", "X axis label"), value=t("Voltage de collision (V)", "Collision voltage (V)"))
        
        if enable_normalization:
            default_y_label = t("Intensité normalisée (%)", "Normalized intensity (%)") if normalization_target_max == 100.0 else t("Intensité normalisée (max=1)", "Normalized intensity (max=1)")
        else:
            default_y_label = t("Survival Yield", "Survival yield")
            
        y_axis_label = st.text_input(t("Nom de l'axe Y", "Y axis label"), value=default_y_label)
        
        col_leg1, col_leg2 = st.columns(2)
        with col_leg1:
            show_legend = st.checkbox(t("Afficher la légende", "Show legend"), value=True)
        with col_leg2:
            survival_curve_name = st.text_input(t("Nom de la courbe Survival Yield", "Survival Yield curve name"), value="Survival Yield")
    
        # Calculs sécurisés des limites par défaut (utiliser les données filtrées pour un meilleur zoom)
        y_default_limit = normalization_target_max if enable_normalization else 1.0
        default_y_min = safe_min(survival_yield_plot, 0.0)
        default_y_max = safe_max(survival_yield_plot, y_default_limit)
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
        breakdown_y_min, breakdown_y_max = validate_axis_limits(breakdown_y_min, breakdown_y_max, 0.0, y_default_limit)
        x_min, x_max = validate_axis_limits(x_min, x_max, default_x_min, default_x_max)
    
        # Graphique Matplotlib AVEC PROTECTION ET FIT SIGMOÏDE
        fig2, ax = plt.subplots()
    
        # Afficher TOUTES les données (y compris les points exclus) mais les distinguer visuellement
        # Points inclus dans l'analyse
        valid_voltages = [v for v, s in zip(voltages_filtered, survival_yield_plot) if np.isfinite(s)]
        valid_survival = [s for s in survival_yield_plot if np.isfinite(s)]
    
        if len(valid_voltages) > 0 and len(valid_survival) > 0:
            ax.plot(valid_voltages, valid_survival, 'o-', label=survival_curve_name, 
                    linewidth=2, markersize=6)
    
        # Points exclus (si il y en a)
        if enable_exclusion and excluded_indices:
            excluded_pairs = [
                (sorted_voltages[i], survival_yield_all_plot[i])
                for i in sorted(excluded_indices)
                if np.isfinite(survival_yield_all_plot[i])
            ]

            if excluded_pairs:
                excluded_voltages_plot = [p[0] for p in excluded_pairs]
                excluded_survival_plot = [p[1] for p in excluded_pairs]
                ax.plot(excluded_voltages_plot, excluded_survival_plot, 'x', 
                       label=f"{survival_curve_name} " + t("(exclus)", "(excluded)"), 
                       color='red', markersize=8, markeredgewidth=2)
    
        # Fragments (données filtrées pour l'analyse)
        for idx in range(num_fragments):
            valid_frag_voltages = [v for v, f in zip(voltages_filtered, fragment_intensities_plot[idx]) if np.isfinite(f)]
            valid_frag_intensities = [f for f in fragment_intensities_plot[idx] if np.isfinite(f)]
    
            if len(valid_frag_voltages) > 0 and len(valid_frag_intensities) > 0:
                ax.plot(valid_frag_voltages, valid_frag_intensities, 'x--', 
                       label=fragment_labels[idx], alpha=0.8)
    
            # Fragments exclus
            if enable_exclusion and excluded_indices:
                excluded_frag_pairs = [
                    (sorted_voltages[i], fragment_intensities_all_plot[idx][i])
                    for i in sorted(excluded_indices)
                    if np.isfinite(fragment_intensities_all_plot[idx][i])
                ]

                if excluded_frag_pairs:
                    excluded_frag_voltages = [p[0] for p in excluded_frag_pairs]
                    excluded_frag_intensities = [p[1] for p in excluded_frag_pairs]
                    ax.plot(excluded_frag_voltages, excluded_frag_intensities, 's', 
                           label=f"{fragment_labels[idx]} " + t("(exclus)", "(excluded)"), 
                           color='red', markersize=6, alpha=0.6)
    
        # Ajouter le fit sigmoïde si activé (utilise les données filtrées)
        if enable_sigmoid and fit_result and fit_result["success"]:
            smooth_data = plot_sigmoid_fit(voltages_filtered, data_to_fit, fit_result)
            if smooth_data:
                v_smooth, y_smooth = smooth_data
                ax.plot(v_smooth, y_smooth, 'r-', linewidth=3,
                       label=f"📈 {fit_result['method']} (V50={fit_result['v50']:.1f}V, R²={fit_result['r2']:.3f})", 
                       alpha=0.8)
    
        if graph_title.strip():
            ax.set_title(graph_title)
            
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
    
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
    
        if show_legend:
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

                    if transition_metrics_result and transition_metrics_result.get("success", False):
                        extra_rows = []
                        if "slope_at_v50" in selected_transition_metric_keys:
                            extra_rows.append({"Metric": "Slope_at_V50", "Value": transition_metrics_result.get("slope_at_v50", np.nan)})
                        if "steepest_slope" in selected_transition_metric_keys:
                            extra_rows.append({"Metric": "Steepest_Slope", "Value": transition_metrics_result.get("steepest_slope_abs", np.nan)})
                            extra_rows.append({"Metric": "Steepest_Slope_Voltage", "Value": transition_metrics_result.get("steepest_voltage", np.nan)})
                        if "v10_v90_width" in selected_transition_metric_keys:
                            extra_rows.append({"Metric": "V10", "Value": transition_metrics_result.get("v10", np.nan)})
                            extra_rows.append({"Metric": "V90", "Value": transition_metrics_result.get("v90", np.nan)})
                            extra_rows.append({"Metric": "V10_V90_Width", "Value": transition_metrics_result.get("v10_v90_width", np.nan)})

                        if extra_rows:
                            df_results = pd.concat([df_results, pd.DataFrame(extra_rows)], ignore_index=True)
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
