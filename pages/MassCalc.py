import streamlit as st
from molmass import Formula
from IsoSpecPy import IsoTotalProb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import io

# =========================
# Dictionnaires de formules
# =========================

AA_FORMULA = {
    'A': 'C3H7NO2', 'R': 'C6H14N4O2', 'N': 'C4H8N2O3',
    'D': 'C4H7NO4', 'C': 'C3H7NO2S', 'Q': 'C5H10N2O3',
    'E': 'C5H9NO4', 'G': 'C2H5NO2', 'H': 'C6H9N3O2',
    'I': 'C6H13NO2', 'L': 'C6H13NO2', 'K': 'C6H14N2O2',
    'M': 'C5H11NO2S', 'F': 'C9H11NO2', 'P': 'C5H9NO2',
    'S': 'C3H7NO3', 'T': 'C4H9NO3', 'W': 'C11H12N2O2',
    'Y': 'C9H11NO3', 'V': 'C5H11NO2',
    # Non canoniques
    'X': 'C5H10N2O3', 'B': 'C4H6N2O3', 'Z': 'C5H8N2O4',
    'J': 'C6H13NO2', 'O': 'C12H21N3O3', 'U': 'C3H7NO2Se'
}

PEPTIDE_MODIFICATIONS = {
    'Phosphorylation (S/T/Y)': 'HPO3',
    'Acétylation (N-term)': 'C2H2O',
    'Méthylation (K)': 'CH2',
    'Ubiquitination (K)': 'C76H113N13O15S',
    'Sumoylation (K)': 'C76H120N20O23S',
    'Oxydation (M)': 'O',
    'Carbamidométhylation (C)': 'C2H3NO'
}
PEPTIDE_MODS_WITH_COUNT = [
    'Phosphorylation (S/T/Y)',
    'Méthylation (K)',
    'Ubiquitination (K)',
    'Sumoylation (K)',
    'Oxydation (M)',
    'Carbamidométhylation (C)'
]
PEPTIDE_MODS_NO_COUNT = [
    'Acétylation (N-term)'
]

NUCLEIC_MODIFICATIONS = {
    'Méthylation (C5)': 'CH2',
    'Thiouridine (sU)': 'S',
    'Pseudouridine (Ψ)': 'C5H4N2O2',
    'Inosine (I)': 'C5H4N4O',
    'Phosphorothioate (pS)': 'custom',
    'Fluorination (2′-F)': 'custom',
    'O-Méthylation (2′-OMe)': 'custom',
    'Phosphorothioate (backbone entier)': 'custom'
}
NUCLEIC_MODS_WITH_COUNT = [
    'Méthylation (C5)',
    'Thiouridine (sU)',
    'Pseudouridine (Ψ)',
    'Phosphorothioate (pS)',
    'Fluorination (2′-F)',
    'O-Méthylation (2′-OMe)',
    'Inosine (I)'
]
NUCLEIC_MODS_NO_COUNT = [
    'Phosphorothioate (backbone entier)'
]

PROTON = 1.007276466812

ADDUCTS = {
    'Na⁺': {'add': 'Na', 'remove': 'H', 'charge': 1},
    'K⁺': {'add': 'K', 'remove': 'H', 'charge': 1},
    'NH₄⁺': {'add': 'NH4', 'remove': 'H', 'charge': 1},
    'Li⁺': {'add': 'Li', 'remove': 'H', 'charge': 1},
    'H⁺': {'add': 'H', 'remove': None, 'charge': 1},
    'Mg²⁺': {'add': 'Mg', 'remove': 'H2', 'charge': 2},
    # Pour négatif : 'Cl⁻': {'add': 'Cl', 'remove': None, 'charge': -1},
}
ADDUCTS_LIST = list(ADDUCTS.keys())

# =========================
# Fonctions utilitaires
# =========================

def clean_formula_input(s):
    cleaned = re.sub(r'[^A-Za-z0-9]', '', s or '')
    return cleaned, cleaned != (s or '')

def clean_sequence_input(s):
    cleaned = re.sub(r'[^A-Za-z]', '', s or '').upper()
    return cleaned, cleaned != (s or '').upper()

def formula_to_html(formula_str):
    def repl(match):
        return match.group(1) + "<sub>" + match.group(2) + "</sub>"
    return re.sub(r'([A-Z][a-z]?)(\d+)', repl, formula_str)

def calculate_peptide_formula(sequence, mod_counts, nterm_mods):
    total = Formula()
    if not sequence:
        return total
    for aa in sequence.upper():
        if aa not in AA_FORMULA:
            raise KeyError(f"Acide aminé non reconnu : {aa}")
        total += Formula(AA_FORMULA[aa])
    total -= Formula(f'H{2*(len(sequence)-1)}O{len(sequence)-1}')
    for mod, count in mod_counts.items():
        total += Formula(PEPTIDE_MODIFICATIONS[mod]) * count
    for mod in nterm_mods:
        total += Formula(PEPTIDE_MODIFICATIONS[mod])
    return total

def calculate_nucleic_formula(sequence, mod_counts, nucleic_type, term_5="OH", term_3="OH", PS=False):
    seq = sequence.upper()
    n = len(seq)
    if nucleic_type == 'ADN':
        bases_dict = {
            'A': 'C10H13N5O3', 'T': 'C10H14N2O5',
            'C': 'C9H13N3O4', 'G': 'C10H13N5O4'
        }
    else:
        bases_dict = {
            'A': 'C10H13N5O4', 'U': 'C9H12N2O6',
            'C': 'C9H13N3O5', 'G': 'C10H13N5O5'
        }
    total = Formula()
    for base in seq:
        if base not in bases_dict:
            raise KeyError(f"Base {base} invalide pour {nucleic_type}")
        total += Formula(bases_dict[base])
    if n > 1:
        total += Formula(f'H{n-1}P{n-1}O{3*(n-1)}')
        total -= Formula(f'H{2*(n-1)}O{n-1}')
    if term_5 == "Phosphate":
        total += Formula("HPO3")
    if term_3 == "Phosphate":
        total += Formula("HPO3")
    elif term_3 == "Phosphate cyclique":
        total += Formula("PO2")
        total -= Formula("H")
    for mod, count in mod_counts.items():
        if mod == 'Phosphorothioate (pS)':
            total -= Formula("O") * count
            total += Formula("S") * count
        elif mod == 'Fluorination (2′-F)':
            if nucleic_type == 'ADN':
                total += Formula("F") * count
                total -= Formula("H") * count
            elif nucleic_type == 'ARN':
                total += Formula("F") * count
                total -= Formula("OH") * count
        elif mod == 'O-Méthylation (2′-OMe)':
            total += Formula("CH3O") * count
        else:
            total += Formula(NUCLEIC_MODIFICATIONS[mod]) * count
    if PS:
        if n > 1:
            total -= Formula(f'O{n-1}')
            total += Formula(f'S{n-1}')
    return total

def count_protons(formula):
    try:
        if not hasattr(formula, 'atoms'):
            return 0
        return int(formula.atoms.get('H', 0))
    except Exception:
        return 0

def apply_adducts_and_charge(formula, adduct_counts, ion_mode, user_charge):
    total = Formula(str(formula.formula))
    total_adduct_charge = 0
    for adduct, count in adduct_counts.items():
        if count <= 0:
            continue
        adduct_info = ADDUCTS[adduct]
        adduct_charge = adduct_info['charge']
        total_adduct_charge += adduct_charge * count
        if adduct_info['add']:
            total += Formula(adduct_info['add']) * count
    if ion_mode == "Positive":
        protons_to_add = user_charge - total_adduct_charge
        if protons_to_add > 0:
            total += Formula('H') * protons_to_add
            for adduct, count in adduct_counts.items():
                if count <= 0:
                    continue
                adduct_info = ADDUCTS[adduct]
                if adduct_info['remove']:
                    if adduct_info['remove'].startswith('H') and len(adduct_info['remove']) > 1:
                        nH = int(adduct_info['remove'][1:])
                        total -= Formula('H') * (nH * count)
                    else:
                        total -= Formula(adduct_info['remove']) * count
        # Si charge apportée uniquement par les adduits, ne rien retirer ni ajouter
    elif ion_mode == "Negative":
        protons_to_remove = user_charge + total_adduct_charge
        if protons_to_remove > 0:
            total -= Formula('H') * protons_to_remove
    return total

def simulate_isotopic_distribution(formula, charge, ion_mode, prob=0.99):
    z = charge if ion_mode == "Positive" else -charge
    iso = IsoTotalProb(formula=str(formula.formula), prob_to_cover=prob)
    masses = np.array(list(iso.masses))
    probs = np.array(list(iso.probs))
    mz_raw = (masses + z * PROTON) / abs(z)
    mono_mz = mz_raw[0]
    mono_decimals = len(f"{mono_mz:.8f}".split('.')[-1].rstrip('0'))
    n = np.round((mz_raw - mono_mz) * abs(z)).astype(int)
    mz_pics = mono_mz + n / abs(z)
    mz_pics = np.round(mz_pics, mono_decimals)
    spacing = 1/abs(z)
    df = pd.DataFrame({'mz': mz_pics, 'intensity': probs, 'mass': masses})
    df_grouped = df.groupby('mz', as_index=False).agg({'intensity': 'sum', 'mass': 'first'})
    df_grouped['intensity'] = df_grouped['intensity'] / df_grouped['intensity'].max() * 100
    return df_grouped, mono_decimals, mono_mz, spacing

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu)/sigma)**2) / (sigma * np.sqrt(2 * np.pi))

def clear_other_fields(changed):
    if changed == "formula_input":
        st.session_state["seq_pep"] = ""
        st.session_state["seq_nuc"] = ""
        cleaned, warn = clean_formula_input(st.session_state["formula_input"])
        st.session_state["formula_input"] = cleaned
        st.session_state["warn_input"] = warn
    elif changed == "seq_pep":
        st.session_state["formula_input"] = ""
        st.session_state["seq_nuc"] = ""
        cleaned, warn = clean_sequence_input(st.session_state["seq_pep"])
        st.session_state["seq_pep"] = cleaned
        st.session_state["warn_input"] = warn
    elif changed == "seq_nuc":
        st.session_state["formula_input"] = ""
        st.session_state["seq_pep"] = ""
        cleaned, warn = clean_sequence_input(st.session_state["seq_nuc"])
        st.session_state["seq_nuc"] = cleaned
        st.session_state["warn_input"] = warn

# =========================
# Interface Streamlit
# =========================

st.title('🧬 Calculateur de Masse exacte & simulation de distribution Isotopique')

simulate_ms = st.checkbox(
    "Afficher la simulation MS (charge, adduits, distribution isotopique)",
    value=True
)

if simulate_ms:
    st.sidebar.header("Paramètres MS")
    ion_mode = st.sidebar.radio("Mode d'ionisation", ["Positive", "Negative"], index=0)
    charge = st.sidebar.number_input(
        "Charge (valeur absolue)",
        min_value=1,
        max_value=100,
        value=1,
        step=1,
        key="charge"
    )
    prob = 0.99
    resolution = st.sidebar.number_input("Résolution (R) à m/z 200", 1000, 1_000_000, 60000, help="R = m/ΔM à m/z 200 (ΔM = largeur de pic à 4σ)")
else:
    ion_mode = "Positive"
    charge = 0
    prob = 0.99
    resolution = 60000

with st.expander("Paramètres des Modifications"):
    peptide_mods = st.multiselect(
        "Modifications peptidiques",
        list(PEPTIDE_MODIFICATIONS.keys()),
        key="peptide_mods"
    )
    peptide_mod_counts = {}
    peptide_nterm_mods = []
    for mod in peptide_mods:
        if mod in PEPTIDE_MODS_WITH_COUNT:
            peptide_mod_counts[mod] = st.number_input(
                f"Nombre de {mod}",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                key=f"count_{mod}"
            )
        elif mod in PEPTIDE_MODS_NO_COUNT:
            peptide_nterm_mods.append(mod)

    nucleic_mods = st.multiselect(
        "Modifications nucléiques",
        list(NUCLEIC_MODIFICATIONS.keys()),
        key="nucleic_mods"
    )
    nucleic_mod_counts = {}
    PS = False
    for mod in nucleic_mods:
        if mod in NUCLEIC_MODS_WITH_COUNT:
            nucleic_mod_counts[mod] = st.number_input(
                f"Nombre de {mod}",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                key=f"count_{mod}"
            )
        elif mod == "Phosphorothioate (backbone entier)":
            PS = True

if simulate_ms:
    with st.expander("Adduits (MS)"):
        selected_adducts = st.multiselect(
            "Sélectionne les adduits à ajouter",
            ADDUCTS_LIST,
            key="adducts"
        )
        adduct_counts = {}
        total_adduct_charge = 0
        max_adducts_dict = {}
        for adduct in selected_adducts:
            adduct_info = ADDUCTS[adduct]
            adduct_charge = adduct_info['charge']
            max_count = (charge - total_adduct_charge) // abs(adduct_charge)
            max_count = max(0, max_count)
            count = st.number_input(
                f"Nombre de {adduct}",
                min_value=0,
                max_value=max_count if max_count > 0 else 0,
                value=0,
                step=1,
                key=f"count_{adduct}"
            )
            adduct_counts[adduct] = count
            total_adduct_charge += count * adduct_charge
else:
    adduct_counts = {}

tab1, tab2, tab3 = st.tabs(["Formule brute", "Peptide/Protéine", "Acide nucléique"])

current_formula = None

def display_formula_and_mass(formula):
    formula_html = formula_to_html(formula.formula)
    nH_final = count_protons(formula)
    if nH_final < 0:
        st.warning(f"⚠️ Trop de protons retirés : la formule finale contient {nH_final} hydrogènes (négatif) !")
    st.markdown(f"**Formule brute :** <span style='font-size:1.3em'>{formula_html}</span>", unsafe_allow_html=True)
    st.subheader(f"Masse exacte (monoisotopique) : {formula.isotope.mass:.8f} Da")
    st.subheader(f"Masse moyenne : {formula.mass:.8f} Da")

with tab1:
    formula_input = st.text_input(
        "Entrez la formule brute (ex: C6H12O6)",
        key="formula_input",
        on_change=clear_other_fields,
        args=("formula_input",)
    )
    if st.session_state.get("warn_input", False):
        st.info("⚠️ Les espaces et caractères non valides ont été supprimés de votre saisie.")
        st.session_state["warn_input"] = False
    if formula_input:
        try:
            current_formula = Formula(formula_input)
            if simulate_ms and (any(adduct_counts.values()) or charge > 0):
                current_formula = apply_adducts_and_charge(current_formula, adduct_counts, ion_mode, charge)
            display_formula_and_mass(current_formula)
        except Exception as e:
            st.error(f"Erreur : {e}")

with tab2:
    seq_pep = st.text_input(
        "Séquence peptidique (1 lettre)",
        key="seq_pep",
        on_change=clear_other_fields,
        args=("seq_pep",)
    )
    if st.session_state.get("warn_input", False):
        st.info("⚠️ Les espaces et caractères non valides ont été supprimés de votre saisie.")
        st.session_state["warn_input"] = False
    if seq_pep:
        try:
            current_formula = calculate_peptide_formula(seq_pep, peptide_mod_counts, peptide_nterm_mods)
            if simulate_ms and (any(adduct_counts.values()) or charge > 0):
                current_formula = apply_adducts_and_charge(current_formula, adduct_counts, ion_mode, charge)
            display_formula_and_mass(current_formula)
        except KeyError as e:
            st.error(str(e))

with tab3:
    nucleic_type = st.radio("Type", ["ADN", "ARN"], key="nucleic_type")
    seq_nuc = st.text_input(
        f"Séquence {nucleic_type} (A, T/U, C, G, I, etc.)",
        key="seq_nuc",
        on_change=clear_other_fields,
        args=("seq_nuc",)
    )
    term_5 = st.selectbox("Terminaison 5'", ["OH", "Phosphate"], key="term_5")
    term_3 = st.selectbox("Terminaison 3'", ["OH", "Phosphate", "Phosphate cyclique"], key="term_3")
    if st.session_state.get("warn_input", False):
        st.info("⚠️ Les espaces et caractères non valides ont été supprimés de votre saisie.")
        st.session_state["warn_input"] = False
    if seq_nuc:
        try:
            current_formula = calculate_nucleic_formula(
                seq_nuc, nucleic_mod_counts, nucleic_type, term_5, term_3, PS
            )
            if simulate_ms and (any(adduct_counts.values()) or charge > 0):
                current_formula = apply_adducts_and_charge(current_formula, adduct_counts, ion_mode, charge)
            display_formula_and_mass(current_formula)
        except KeyError as e:
            st.error(str(e))

if simulate_ms and current_formula and charge > 0:
    try:
        df_grouped, decimals, mono_mz, spacing = simulate_isotopic_distribution(current_formula, charge, ion_mode, prob)
        resolution_effective = resolution * np.sqrt(mono_mz / 200)
        required_resolution = mono_mz / spacing
        sigma = mono_mz / (4 * resolution_effective)
        if resolution_effective < required_resolution:
            st.warning(f"Résolution insuffisante pour séparer les pics isotopiques (R requis : {required_resolution:.0f}, R effectif : {resolution_effective:.0f})")
        mz_min = df_grouped['mz'].min() - 10 * sigma
        mz_max = df_grouped['mz'].max() + 10 * sigma
        mz_grid = np.linspace(mz_min, mz_max, 10000)
        intensity = np.zeros_like(mz_grid)
        for _, row in df_grouped.iterrows():
            mz = row['mz']
            proba = row['intensity']
            intensity += proba * gaussian(mz_grid, mz, sigma)
        intensity = (intensity / intensity.max()) * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(mz_grid, intensity, color='blue', linewidth=1)
        ax.set_title(f'{current_formula.formula} | charge: z={charge} | R={resolution} (σ={sigma:.5f} Da)\nDistribution isotopique ({ion_mode})', fontsize=13)
        ax.set_xlabel('m/z', fontsize=12)
        ax.set_ylabel('Intensité relative (%)', fontsize=12)
        if st.checkbox("Afficher les pics théoriques (bâtonnets)"):
            ax.stem(df_grouped['mz'], df_grouped['intensity'], linefmt='grey', markerfmt=' ', basefmt=' ')
        st.pyplot(fig)

        svg_buffer = io.BytesIO()
        fig.savefig(svg_buffer, format='svg')
        svg_buffer.seek(0)
        st.download_button(
            label="📥 Exporter le graphique en SVG",
            data=svg_buffer,
            file_name="distribution_isotopique.svg",
            mime="image/svg+xml"
        )

        df_display = df_grouped.rename(columns={'mz': 'm/z', 'intensity': 'Intensité (%)', 'mass': 'Masse exacte (Da)'})
        st.subheader("📊 Détail des pics isotopiques")
        st.dataframe(
            df_display.style.format({
                'm/z': f'{{:.{decimals}f}}',
                'Intensité (%)': '{:.2f}%',
                'Masse exacte (Da)': '{:.8f}'
            }),
            height=400,
            use_container_width=True
        )
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exporter en CSV",
            data=csv,
            file_name='isotopes.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"Erreur de simulation : {str(e)}")

with st.expander("Aide & Astuces"):
    st.markdown("""
- **Case à cocher pour activer/désactiver la simulation MS (charges, adduits, distribution isotopique)**.
- **Si décochée, seule la masse exacte/moyenne de la molécule neutre est affichée.**
- **Champs réinitialisés automatiquement dès que tu modifies un champ.**
- **Caractères invalides et espaces supprimés automatiquement.**
- **Formule brute affichée avec indices chimiques.**
- **Terminaisons 5'/3' personnalisables pour les acides nucléiques (OH, phosphate, phosphate cyclique).**
- **Formule brute des acides nucléiques corrigée (nucléosides, squelette, H/O, terminaisons).**
- **Pics gaussiens** : largeur calculée automatiquement à partir de la résolution instrumentale (R) et de la charge.
- **Si la résolution effective est insuffisante, les pics isotopiques ne seront pas séparés (avertissement affiché).**
- **Export CSV/SVG** : toutes les données isotopiques et le graphique sont exportables.
- **Adduits** : possibilité d'ajouter plusieurs adduits (Na⁺, K⁺, Mg²⁺, etc.) et de spécifier leur nombre, avec correction automatique des protons et limitation stricte à la charge totale.
- **Le nombre de protons (H) est automatiquement ajusté selon la charge sélectionnée, même sans adduit, pour simuler [M+zH]^z+ ou [M–zH]^z–.
    """)
