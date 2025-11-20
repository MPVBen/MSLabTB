import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from scipy.special import gamma
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

def log_sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))

def log_sigmoid_derivative(x, x0, k):
    return k * np.exp(-k * (x - x0)) / (1 + np.exp(-k * (x - x0)))**2

def equation_to_solve(T, E, theta_values, s_global):
    C_value = sum((theta / T) / (np.exp(theta / T) - 1) for theta in theta_values) / s_global
    s_corrected = C_value * s_global
    return E - s_corrected * k_B * T

def calculate_temperature(avg_E, theta_values, s_global):
    initial_guess = 600
    T_solution = fsolve(equation_to_solve, initial_guess, args=(avg_E, theta_values, s_global))
    return T_solution[0]

def s_correction(T, theta_values, s_global):
    C_value = sum((theta / T) / (np.exp(theta / T) - 1) for theta in theta_values) / s_global
    s_corrected = C_value * s_global
    return s_corrected

def calculate_error_in_T(sigma_E, T, theta_values):
    sum_term_1 = np.sum([theta / T / (np.exp(theta / T) - 1) for theta in theta_values])
    sum_term_2 = np.sum([(theta / T**2) * (theta / T * np.exp(theta / T) - (np.exp(theta / T) - 1)) / 
                        (np.exp(theta / T) - 1)**2 for theta in theta_values])
    dEint_dT = k_B * (sum_term_1 + T * sum_term_2)
    sigma_T = sigma_E / abs(dEint_dT) / 4  # facteur empirique pour être raccord avec le code original
    return sigma_T

def fit_data(x_data, y_data):
    popt, pcov = curve_fit(log_sigmoid, x_data, y_data, maxfev=10000)
    avg_E = popt[0]
    sigma_E = np.sqrt(pcov[0, 0])
    predicted_values = log_sigmoid(x_data, *popt)
    residuals = y_data - predicted_values
    sse = np.sum(residuals**2)
    y_mean = np.mean(y_data)
    total_variance = np.sum((y_data - y_mean)**2)
    r_square = 1 - (sse / total_variance)
    return popt, sigma_E, r_square

st.title("🌡️ Calculateur de Température Effective en LDI-MS")

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
            st.warning(f"⚠️ Réplicat {i+1}: Vous devez entrer exactement 4 valeurs comprises entre 0 et 1 (bornes incluses)!")
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

    # Charger tous les fichiers de fréquences
    freq_files = []
    for file in uploaded_files:
        freq_files.append({'name': file.name, 'lines': file.getvalue().decode('utf-8').splitlines()})

    # Préparer les theta_values pour chaque fichier
    freq_info = []
    for f in freq_files:
        match = re.search(r'_(\d+)\.txt$', f['name'])
        if not match:
            st.error(f"Nom de fichier incorrect : {f['name']}")
            continue
        s_global = int(match.group(1))
        theta_values = [float(line) * h * c_light / k_B for line in f['lines'] if line.strip()]
        freq_info.append({'name': f['name'], 's_global': s_global, 'theta_values': theta_values})

    # Calculs pour chaque réplicat
    results = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    x_fit = np.linspace(0, 5, 200)
    for i, (y_data, label) in enumerate(zip(replicats, labels)):
        popt, sigma_E, r2 = fit_data(x_data, y_data)
        x0, k = popt
        Ts, sigmas_T = [], []
        for freq in freq_info:
            T = calculate_temperature(x0, freq['theta_values'], freq['s_global'])
            sigma_T = calculate_error_in_T(sigma_E, T, freq['theta_values'])
            Ts.append(T)
            sigmas_T.append(sigma_T)
        T_mean = np.mean(Ts)
        sigma_T_mean = np.mean(sigmas_T)
        # Tracé avec la couleur choisie
        color = colors[i]
        y_fit = log_sigmoid(x_fit, x0, k)
        y_deriv = log_sigmoid_derivative(x_fit, x0, k)
        ax1.plot(x_fit, y_fit, label=label, color=color)
        ax1.scatter(x_data, y_data, color=color, edgecolor='black')
        ax2.plot(x_fit, y_deriv, label=label, color=color)
        # Résultats
        st.write(f"{label}")
        st.write(f"{y_data}")
        st.write(f"T = {T_mean:.2f} ± {sigma_T_mean:.2f}, <E_int> = {x0:.2f} ± {sigma_E:.2f}, R²= {r2:.3f}")
        results.append({
            'Réplicat': label,
            'T (K)': f"{T_mean:.2f} ± {sigma_T_mean:.2f}",
            '<E_int> (eV)': f"{x0:.2f} ± {sigma_E:.2f}",
            'R²': f"{r2:.3f}"
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
