import streamlit as st
import pandas as pd
import math

# ============================================================================
# FONCTIONS UTILITAIRES (CONVERSIONS)
# ============================================================================

# Facteurs de conversion vers l'unité de base (Molaire, Litre, Gramme)
CONV_CONC_MOLAR = {"µM": 1e-6, "mM": 1e-3, "M": 1.0}
CONV_VOL = {"nL": 1e-9, "µL": 1e-6, "mL": 1e-3, "L": 1.0}
CONV_MASS = {"ng": 1e-9, "µg": 1e-6, "mg": 1e-3, "g": 1.0, "kg": 1000.0}

def get_base_value(value, unit, type_measure="conc_molar"):
    """Convertit une valeur vers son unité SI de base pour les calculs."""
    if type_measure == "conc_molar" and unit in CONV_CONC_MOLAR:
        return value * CONV_CONC_MOLAR[unit]
    elif type_measure == "vol" and unit in CONV_VOL:
        return value * CONV_VOL[unit]
    elif type_measure == "mass" and unit in CONV_MASS:
        return value * CONV_MASS[unit]
    # Cas spécifiques pour concentration massique (base calcul = g/L)
    elif type_measure == "conc_mass":
        if unit == "g/L" or unit == "mg/mL": return value
        if unit == "µg/mL": return value * 1e-3
        if unit == "ng/µL": return value * 1e-3
    return None

def app():
    st.set_page_config(page_title="LabCalc", page_icon="🧮")
    st.title("🧮 LabCalc - Calculateurs de Laboratoire (Corrigé)")
    st.markdown("---")
    
    # Menu de sélection
    calculator_type = st.selectbox(
        "Choisissez un calculateur",
        [
            "💧 Dilution (C1V1 = C2V2)",
            "⚖️ Masse à peser",
            "📊 Préparation de solution mère",
            "🔄 Conversion d'unités",
            "📐 Normalité et Molarité",
            "🧪 pH et tampons",
            "💉 Volume d'injection",
            "📈 Facteur de dilution série"
        ]
    )
    
    st.markdown("---")
    
    # ============================================================================
    # 1. CALCULATEUR DE DILUTION (CORRIGÉ)
    # ============================================================================
    if calculator_type == "💧 Dilution (C1V1 = C2V2)":
        st.subheader("💧 Calculateur de Dilution")
        st.info("Correction appliquée : Les unités sont maintenant converties avant le calcul.")
        
        col1, col2 = st.columns(2)
        
        # Listes d'unités compatibles
        units_molar = ["M", "mM", "µM"]
        units_mass = ["g/L", "mg/mL", "µg/mL"]
        all_units = units_molar + units_mass
        
        with col1:
            st.markdown("**📥 Solution initiale (stock)**")
            c1 = st.number_input("Concentration initiale (C₁)", min_value=0.0, value=5.0, step=0.1, format="%.2f")
            unit_c1 = st.selectbox("Unité C₁", all_units, index=0, key="unit_c1") # Par défaut M
            
        with col2:
            st.markdown("**📤 Solution finale (diluée)**")
            c2 = st.number_input("Concentration finale (C₂)", min_value=0.0, value=150.0, step=0.1, format="%.2f")
            unit_c2 = st.selectbox("Unité C₂", all_units, index=1, key="unit_c2") # Par défaut mM
        
        st.markdown("**🎯 Volume souhaité**")
        v2 = st.number_input("Volume final (V₂)", min_value=0.0, value=200.0, step=10.0)
        unit_v2 = st.selectbox("Unité V₂", ["µL", "mL", "L"], key="unit_v2")
        
        if st.button("Calculer la dilution", key="calc_dilution"):
            if c1 > 0 and c2 > 0 and v2 > 0:
                # 1. Vérifier si on compare des pommes avec des pommes (Molaire vs Massique)
                is_c1_molar = unit_c1 in units_molar
                is_c2_molar = unit_c2 in units_molar
                
                if is_c1_molar != is_c2_molar:
                    st.error("⛔ Erreur : Vous essayez de diluer une concentration Molaire vers une Massique (ou inversement). Impossible sans la Masse Molaire.")
                else:
                    # 2. Normaliser les concentrations pour le calcul
                    type_c = "conc_molar" if is_c1_molar else "conc_mass"
                    c1_norm = get_base_value(c1, unit_c1, type_c)
                    c2_norm = get_base_value(c2, unit_c2, type_c)
                    
                    # 3. Vérifier que Stock > Finale
                    if c2_norm >= c1_norm:
                        st.error(f"⛔ Erreur de logique : La solution finale ({c2} {unit_c2}) est plus concentrée que le stock ({c1} {unit_c1}) !")
                    else:
                        # 4. Calcul : V1 = (C2 * V2) / C1
                        # On garde V2 dans son unité d'origine pour que V1 sorte dans la même unité
                        v1 = (c2_norm * v2) / c1_norm
                        volume_solvant = v2 - v1
                        facteur_dilution = c1_norm / c2_norm
                        
                        st.success("✅ Résultats corrects :")
                        
                        r1, r2, r3 = st.columns(3)
                        with r1: st.metric("Volume à prélever (V₁)", f"{v1:.2f} {unit_v2}")
                        with r2: st.metric("Volume de solvant", f"{volume_solvant:.2f} {unit_v2}")
                        with r3: st.metric("Facteur", f"1 : {facteur_dilution:.1f}")
                        
                        st.info(f"📋 **Protocole :** Prélever **{v1:.2f} {unit_v2}** de solution stock et ajouter **{volume_solvant:.2f} {unit_v2}** de solvant.")
            else:
                st.error("❌ Les valeurs doivent être supérieures à 0")
    
    # ============================================================================
    # 2. MASSE À PESER (CORRIGÉ)
    # ============================================================================
    elif calculator_type == "⚖️ Masse à peser":
        st.subheader("⚖️ Calculateur de Masse à Peser")
        
        col1, col2 = st.columns(2)
        with col1:
            masse_molaire = st.number_input("Masse molaire (g/mol)", min_value=0.0, value=180.16)
            concentration = st.number_input("Concentration", min_value=0.0, value=10.0)
            unit_conc = st.selectbox("Unité", ["mM", "µM", "M", "mg/mL", "g/L"])
        with col2:
            volume = st.number_input("Volume", min_value=0.0, value=50.0)
            unit_vol = st.selectbox("Unité Vol", ["mL", "µL", "L"])
            purete = st.number_input("Pureté (%)", value=100.0)
        
        if st.button("Calculer masse"):
            # Conversion Volume -> Litres
            vol_L = get_base_value(volume, unit_vol, "vol")
            masse_g = 0
            
            # Calcul Masse théorique en grammes
            if unit_conc in ["M", "mM", "µM"]:
                conc_M = get_base_value(concentration, unit_conc, "conc_molar")
                masse_g = conc_M * vol_L * masse_molaire
            else:
                # mg/mL = g/L. Le bug original multipliait par 1000 en trop ici.
                conc_g_L = get_base_value(concentration, unit_conc, "conc_mass")
                masse_g = conc_g_L * vol_L
            
            # Correction Pureté
            masse_reelle = masse_g / (purete / 100.0)
            
            # Affichage intelligent (mg ou g)
            if masse_reelle < 1e-3:
                st.metric("Masse à peser", f"{masse_reelle*1e6:.2f} µg")
            elif masse_reelle < 1:
                st.metric("Masse à peser", f"{masse_reelle*1e3:.2f} mg")
            else:
                st.metric("Masse à peser", f"{masse_reelle:.4f} g")

    # ============================================================================
    # 3. PRÉPARATION DE SOLUTION MÈRE (CORRIGÉ)
    # ============================================================================
    elif calculator_type == "📊 Préparation de solution mère":
        st.subheader("📊 Préparation de Solution Mère")
        col1, col2 = st.columns(2)
        with col1:
            masse = st.number_input("Masse dispo (mg)", value=10.0)
            mw = st.number_input("Masse molaire (g/mol)", value=300.0)
        with col2:
            conc_cible = st.number_input("Conc. cible", value=10.0)
            unit_cible = st.selectbox("Unité", ["mM", "M", "mg/mL"])
            
        if st.button("Calculer volume solvant"):
            masse_g = masse / 1000.0
            vol_L = 0
            
            if unit_cible == "mg/mL":
                # C = m/V -> V = m/C
                # (g) / (g/L) = L
                conc_g_L = conc_cible # car mg/mL = g/L
                vol_L = masse_g / conc_g_L
            else:
                # C = n/V = (m/MW)/V -> V = m / (MW*C)
                conc_M = get_base_value(conc_cible, unit_cible, "conc_molar")
                vol_L = masse_g / (mw * conc_M)
                
            st.metric("Volume de solvant à ajouter", f"{vol_L*1e3:.2f} mL")

    # ============================================================================
    # 4. CONVERSION D'UNITÉS
    # ============================================================================
    elif calculator_type == "🔄 Conversion d'unités":
        st.subheader("🔄 Convertisseur")
        type_conv = st.radio("Type", ["Masse", "Volume", "Concentration"])
        
        c1, c2, c3 = st.columns(3)
        with c1: val = st.number_input("Valeur", value=1.0)
        
        if type_conv == "Masse":
            with c2: u1 = st.selectbox("De", list(CONV_MASS.keys()))
            with c3: u2 = st.selectbox("Vers", list(CONV_MASS.keys()))
            res = (val * CONV_MASS[u1]) / CONV_MASS[u2]
            
        elif type_conv == "Volume":
            with c2: u1 = st.selectbox("De", list(CONV_VOL.keys()))
            with c3: u2 = st.selectbox("Vers", list(CONV_VOL.keys()))
            res = (val * CONV_VOL[u1]) / CONV_VOL[u2]
            
        elif type_conv == "Concentration":
            with c2: u1 = st.selectbox("De", list(CONV_CONC_MOLAR.keys()))
            with c3: u2 = st.selectbox("Vers", list(CONV_CONC_MOLAR.keys()))
            res = (val * CONV_CONC_MOLAR[u1]) / CONV_CONC_MOLAR[u2]
            
        st.success(f"Resultat : {res:.6g} {u2}")

    # ============================================================================
    # 5. NORMALITÉ
    # ============================================================================
    elif calculator_type == "📐 Normalité et Molarité":
        st.subheader("📐 Normalité (N) ↔ Molarité (M)")
        c1, c2 = st.columns(2)
        with c1: 
            M = st.number_input("Molarité (M)", value=1.0)
            eq = st.number_input("Équivalents (ex: 2 pour H2SO4)", value=1, min_value=1)
            st.metric("Normalité", f"{M * eq:.2f} N")
        with c2:
            N = st.number_input("Normalité (N)", value=1.0)
            eq2 = st.number_input("Équivalents", value=1, min_value=1, key="eq2")
            st.metric("Molarité", f"{N / eq2:.2f} M")

    # ============================================================================
    # 6. pH (SIMPLE)
    # ============================================================================
    elif calculator_type == "🧪 pH et tampons":
        st.subheader("🧪 Calculateur pH")
        mode = st.radio("Mode", ["Acide Fort", "Henderson-Hasselbalch"])
        if mode == "Acide Fort":
            c = st.number_input("Concentration (M)", value=0.01, format="%.4f")
            if c > 0: st.metric("pH", f"{-math.log10(c):.2f}")
        else:
            pka = st.number_input("pKa", value=4.76)
            ratio = st.number_input("Ratio [A-]/[HA]", value=1.0)
            if ratio > 0: st.metric("pH", f"{pka + math.log10(ratio):.2f}")

    # ============================================================================
    # 7. VOLUME D'INJECTION (RÉPARÉ, n'était pas implémenté)
    # ============================================================================
    elif calculator_type == "💉 Volume d'injection":
        st.subheader("💉 Volume d'injection (In Vivo)")
        st.markdown("Calcule quel volume injecter à une souris/rat pour donner une dose précise.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 1. Solution Stock")
            conc = st.number_input("Concentration", value=2.0)
            u_conc = st.selectbox("Unité Stock", ["mg/mL", "µg/mL", "mM"])
            if u_conc == "mM":
                mw = st.number_input("Masse Molaire (g/mol)", value=300.0)
        
        with c2:
            st.markdown("##### 2. Dose Cible")
            dose = st.number_input("Quantité à injecter", value=10.0)
            u_dose = st.selectbox("Unité Dose", ["mg", "µg", "kg (dose pondérale)"])
            
            poids = 0
            if "kg" in u_dose:
                poids = st.number_input("Poids animal (g)", value=25.0)
                dose_absolue = (dose * (poids/1000.0)) # mg si dose en mg/kg
                u_dose_calc = "mg"
            else:
                dose_absolue = dose
                u_dose_calc = u_dose

        if st.button("Calculer Volume"):
            # Conversion tout en mg et mL
            conc_mg_mL = 0
            if u_conc == "mg/mL": conc_mg_mL = conc
            elif u_conc == "µg/mL": conc_mg_mL = conc / 1000.0
            elif u_conc == "mM": conc_mg_mL = conc * mw / 1000.0
            
            dose_mg = 0
            if u_dose_calc == "mg": dose_mg = dose_absolue
            elif u_dose_calc == "µg": dose_mg = dose_absolue / 1000.0
            
            if conc_mg_mL > 0:
                vol_mL = dose_mg / conc_mg_mL
                vol_uL = vol_mL * 1000.0
                st.success(f"✅ Injecter : **{vol_uL:.2f} µL**")
            else:
                st.error("Concentration invalide")

    # ============================================================================
    # 8. DILUTIONS SÉRIE
    # ============================================================================
    elif calculator_type == "📈 Facteur de dilution série":
        st.subheader("📈 Gamme de Calibration")
        start = st.number_input("Conc. Départ", value=100.0)
        factor = st.number_input("Facteur de dilution", value=2.0)
        n = st.number_input("Nombre de points", value=8, step=1)
        
        if st.button("Générer"):
            df = pd.DataFrame([{"Point": i+1, "Concentration": start/(factor**i)} for i in range(int(n))])
            st.dataframe(df)

if __name__ == "__main__":
    app()
