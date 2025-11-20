import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from scipy.special import gamma, gammainc
import re

# Constantes physiques
k_B = 8.617e-5  # eV/K
c_light = 2.998e10  # cm/s
h = 4.1356677e-15  # eV/Hz

# Energies de dissociation fixes
E_0_pOCH3 = 1.57945884614051
E_0_pCH3 = 1.94965647686148
E_0_pF = 2.11588903948509
E_0_pCN = 2.42763302064663
x_data = [0, E_0_pOCH3, E_0_pCH3, E_0_pF, E_0_pCN, 5]

# Fonctions analytiques
def s_correction(T, theta_values, s_global):
    C_value = sum((theta / T) / (np.exp(theta / T) - 1) for theta in theta_values) / s_global
    return C_value * s_global

def s_correction_and_derivative(T, theta_values, s_global):
    C_value = sum((theta / T) / (np.exp(theta / T) - 1) for theta in theta_values) / s_global
    s_corrected = C_value * s_global
    ds_corrected_dT = sum(
        -(theta / T ** 2) * (np.exp(theta / T) / (np.exp(theta / T) - 1) ** 2)
        for theta in theta_values
    ) / s_global
    return s_corrected, ds_corrected_dT

def F_analytical(E, T, theta_values, s_global):
    s_corr = s_correction_and_derivative(T, theta_values, s_global)[0]
    return gammainc(s_corr, E / (k_B * T))

def fit_experimental_data(E_exp, F_exp, theta_values, s_global, p0):
    popt, pcov = curve_fit(
        lambda E, T: F_analytical(E, T, theta_values, s_global),
        E_exp, F_exp, p0
    )
    return popt, pcov

def calculate_error_in_E(s_corrected, T_fit, sigma_T, sigma_s, ds_corrected_dT):
    dE_dT = (s_corrected + T_fit * ds_corrected_dT) * k_B
    dE_ds = k_B * T_fit
    sigma_E = np.sqrt((dE_dT * sigma_T) ** 2 + (dE_ds * sigma_s) ** 2)
    return sigma_E

def average_y_data(x_data, y_data):
    x_array = np.array(x_data)
    y_array = np.array(y_data)
    unique_x, indices = np.unique(x_array, return_inverse=True)
    averaged_y = np.array([y_array[indices == i].mean() for i in range(len(unique_x))])
    return averaged_y

# Interface Streamlit
st.title("🌡️ Calculateur de Température Effective (fit gamma incomplète)")

st.header("📊 Entrée des survival yields")

st.info(
    "Veuillez entrer **4 valeurs de survival yield** (comprises entre 0 et 1, bornes incluses) correspondant aux énergies intermédiaires. "
    "Les bornes 0 et 1 seront automatiquement ajoutées. "
    "Exemple : `0.19, 0.64, 0.62, 0.69`"
)

num_replicats = st.number_input("Nombre de réplicats", min_value=1, max_value=10, value=4)
replicats = []
labels = []
colors = []  # Liste pour stocker les couleurs choisies

for i in range(num_replicats):
    col1, col2, col3 = st.columns([4, 2, 1])  # Ajout d'une colonne pour la couleur
    with col1:
        y_input = st.text_input(
            f"Survival yields (réplicat {i+1}) (4 valeurs, séparées par des virgules)",
            value="",
            key=f"sy_input_{i}"
        )
    with col2:
        label = st.text_input(f"Label (réplicat {i+1})", value=f"Réplicat {i+1}", key=f"label_{i}")
    with col3:
        # Palette de couleurs par défaut
        default_colors = ['#48A97B', '#3E8625', '#9AC72D', '#FDDA0D', '#FDCB00', '#D62728']
        color = st.color_picker(f"Couleur {i+1}", value=default_colors[i % len(default_colors)], key=f"color_{i}")
    
    try:
        y_values = [float(v.strip()) for v in y_input.split(',') if v.strip() != ""]
        if len(y_values) == 4 and all(0 <= v <= 1 for v in y_values):
            # Ajoute les bornes 0 et 1 automatiquement
            y_full = [0.0] + y_values + [1.0]
            replicats.append(y_full)
            labels.append(label)
            colors.append(color)
        elif y_input:
            st.warning(f"⚠️ Réplicat {i+1}: Vous devez entrer exactement 4 valeurs entre 0 et 1!")
    except:
        if y_input:
            st.error(f"❌ Format invalide pour le réplicat {i+1}!")

st.header("📁 Fichiers de fréquences vibratoires")
uploaded_files = st.file_uploader(
    "Sélectionnez vos fichiers de fréquences",
    type=['txt'],
    accept_multiple_files=True
)

if st.button("🚀 Calculer les températures effectives"):
    if not replicats:
        st.error("❌ Veuillez entrer au moins un réplicat valide!")
        st.stop()
    if not uploaded_files:
        st.error("❌ Aucun fichier de fréquences n'a été téléversé. Veuillez charger au moins un fichier de fréquences pour effectuer le calcul.")
        st.stop()

    # Préparer les theta_values pour chaque fichier
    freq_info = []
    for file in uploaded_files:
        match = re.search(r'_(\d+)\.txt$', file.name)
        if not match:
            st.error(f"Nom de fichier incorrect : {file.name}")
            continue
        s_global = int(match.group(1))
        freq_lines = file.getvalue().decode('utf-8').splitlines()
        theta_values = [float(line) * h * c_light / k_B for line in freq_lines if line.strip()]
        freq_info.append({'name': file.name, 's_global': s_global, 'theta_values': theta_values})

    results = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    x_fit = np.linspace(0, 5, 200)

    for i, (y_data, label) in enumerate(zip(replicats, labels)):
        Ts, sigmas_T, s_corrs, avg_Es, sigma_Es, R2s = [], [], [], [], [], []
        # On ne fit que sur les points internes (pas les bornes 0 et 1)
        x_fitdata = np.array(x_data[1:-1])
        y_fitdata = np.array(y_data[1:-1])
        for freq in freq_info:
            # Fit
            p0 = [800]
            popt, pcov = fit_experimental_data(x_fitdata, y_fitdata, freq['theta_values'], freq['s_global'], p0)
            T_fit = popt[0]
            sigma_T = np.sqrt(pcov[0][0]) if pcov.shape == (1, 1) else 0.0
            s_corr, ds_corr_dT = s_correction_and_derivative(T_fit, freq['theta_values'], freq['s_global'])
            sigma_s = ds_corr_dT * sigma_T
            avg_E = s_corr * k_B * T_fit
            sigma_E = calculate_error_in_E(s_corr, T_fit, sigma_T, sigma_s, ds_corr_dT)
            # R²
            predicted = gammainc(s_corr, x_fitdata / (k_B * T_fit))
            residuals = y_fitdata - predicted
            sse = np.sum(residuals**2)
            y_mean = np.mean(y_fitdata)
            total_variance = np.sum((y_fitdata - y_mean)**2)
            r2 = 1 - (sse / total_variance)
            # Stockage
            Ts.append(T_fit)
            sigmas_T.append(sigma_T)
            s_corrs.append(s_corr)
            avg_Es.append(avg_E)
            sigma_Es.append(sigma_E)
            R2s.append(r2)
        # Moyennes sur tous les fichiers
        T_mean = np.mean(Ts)
        sigma_T_mean = np.mean(sigmas_T)
        s_corr_mean = np.mean(s_corrs)
        avg_E_mean = np.mean(avg_Es)
        sigma_E_mean = np.mean(sigma_Es)
        R2_mean = np.mean(R2s)
        # Tracé avec la couleur choisie
        color = colors[i]  # Utilisation de la couleur spécifique au réplicat
        y_gammainc = gammainc(s_corr_mean, np.array(x_fit) / (k_B * T_mean))
        y_deriv = np.gradient(y_gammainc, x_fit)
        ax1.plot(x_fit, y_gammainc, label=label, color=color)
        ax1.scatter(x_data, y_data, color=color, edgecolor='black')
        ax2.plot(x_fit, y_deriv, label=label, color=color)
        # Résultats
        st.write(f"{label}")
        st.write(f"{y_data}")
        st.write(f"T = {T_mean:.2f} ± {sigma_T_mean:.2f}, <E_int> = {avg_E_mean:.2f} ± {sigma_E_mean:.2f}, R²= {R2_mean:.3f}")
        results.append({
            'Réplicat': label,
            'T (K)': f"{T_mean:.2f} ± {sigma_T_mean:.2f}",
            '<E_int> (eV)': f"{avg_E_mean:.2f} ± {sigma_E_mean:.2f}",
            'R²': f"{R2_mean:.3f}"
        })
    ax1.set_xlabel('Énergie de dissociation (eV)')
    ax1.set_ylabel('Survival Yield')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.set_xlabel('Énergie de dissociation (eV)')
    ax2.set_ylabel('Dérivée')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig)
    st.header("📋 Résumé des résultats")
    st.dataframe(pd.DataFrame(results))
    csv = pd.DataFrame(results).to_csv(index=False).encode('utf-8')
    st.download_button(
        "💾 Télécharger les résultats (CSV)",
        csv,
        "resultats_temperature.csv",
        "text/csv"
    )
