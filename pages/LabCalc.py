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
    st.title("🧮 LabCalc - Calculateurs de Laboratoire")
    st.markdown("---")
    
    # Menu de sélection
    calculator_type = st.selectbox(
        "Choisissez un calculateur",
        [
            "💧 Dilution (C1V1 = C2V2)",
            "⚖️ Masse à peser",
            "📊 Préparation de solution mère",
            "🔄 Conversion d'unités (Avancé)",
            "🧪 pH et tampons",
            "📈 Facteur de dilution série"
        ]
    )
    
    st.markdown("---")
    
    # ============================================================================
    # 1. CALCULATEUR DE DILUTION
    # ============================================================================
    if calculator_type == "💧 Dilution (C1V1 = C2V2)":
        st.subheader("💧 Calculateur de Dilution")
        
        col1, col2 = st.columns(2)
        units_molar = ["M", "mM", "µM"]
        units_mass = ["g/L", "mg/mL", "µg/mL"]
        all_units = units_molar + units_mass
        
        with col1:
            st.markdown("**📥 Solution initiale (stock)**")
            c1 = st.number_input("Concentration initiale (C₁)", min_value=0.0, value=5.0, step=0.1, format="%.2f")
            unit_c1 = st.selectbox("Unité C₁", all_units, index=0, key="unit_c1")
            
        with col2:
            st.markdown("**📤 Solution finale (diluée)**")
            c2 = st.number_input("Concentration finale (C₂)", min_value=0.0, value=150.0, step=0.1, format="%.2f")
            unit_c2 = st.selectbox("Unité C₂", all_units, index=1, key="unit_c2")
        
        st.markdown("**🎯 Volume souhaité**")
        v2 = st.number_input("Volume final (V₂)", min_value=0.0, value=200.0, step=10.0)
        unit_v2 = st.selectbox("Unité V₂", ["µL", "mL", "L"], key="unit_v2")
        
        if st.button("Calculer la dilution", key="calc_dilution"):
            if c1 > 0 and c2 > 0 and v2 > 0:
                is_c1_molar = unit_c1 in units_molar
                is_c2_molar = unit_c2 in units_molar
                
                if is_c1_molar != is_c2_molar:
                    st.error("⛔ Erreur : Incompatibilité Molaire/Massique sans masse molaire.")
                else:
                    type_c = "conc_molar" if is_c1_molar else "conc_mass"
                    c1_norm = get_base_value(c1, unit_c1, type_c)
                    c2_norm = get_base_value(c2, unit_c2, type_c)
                    
                    if c2_norm >= c1_norm:
                        st.error(f"⛔ Erreur : La solution finale est plus concentrée que le stock !")
                    else:
                        v1 = (c2_norm * v2) / c1_norm
                        volume_solvant = v2 - v1
                        facteur_dilution = c1_norm / c2_norm
                        
                        st.success("✅ Résultats :")
                        r1, r2, r3 = st.columns(3)
                        with r1: st.metric("Volume à prélever (V₁)", f"{v1:.2f} {unit_v2}")
                        with r2: st.metric("Volume de solvant", f"{volume_solvant:.2f} {unit_v2}")
                        with r3: st.metric("Facteur", f"1 : {facteur_dilution:.1f}")
                        st.info(f"📋 **Protocole :** Prélever **{v1:.2f} {unit_v2}** de stock et ajouter **{volume_solvant:.2f} {unit_v2}** de solvant.")
            else:
                st.error("❌ Les valeurs doivent être supérieures à 0")
    
    # ============================================================================
    # 2. MASSE À PESER
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
            vol_L = get_base_value(volume, unit_vol, "vol")
            masse_g = 0
            
            if unit_conc in ["M", "mM", "µM"]:
                conc_M = get_base_value(concentration, unit_conc, "conc_molar")
                masse_g = conc_M * vol_L * masse_molaire
            else:
                conc_g_L = get_base_value(concentration, unit_conc, "conc_mass")
                masse_g = conc_g_L * vol_L
            
            masse_reelle = masse_g / (purete / 100.0)
            
            if masse_reelle < 1e-3:
                st.metric("Masse à peser", f"{masse_reelle*1e6:.2f} µg")
            elif masse_reelle < 1:
                st.metric("Masse à peser", f"{masse_reelle*1e3:.2f} mg")
            else:
                st.metric("Masse à peser", f"{masse_reelle:.4f} g")

    # ============================================================================
    # 3. PRÉPARATION DE SOLUTION MÈRE
    # ============================================================================
    elif calculator_type == "📊 Préparation de solution mère":
        st.subheader("📊 Préparation de Solution Mère")
        
        col1, col2 = st.columns(2)
        with col1:
            masse = st.number_input("Masse disponible (mg)", value=5.0, min_value=0.0)
            mw = st.number_input("Masse molaire (g/mol)", value=300.0, min_value=0.0)
        with col2:
            conc_cible = st.number_input("Concentration cible", value=10.0, min_value=0.0)
            unit_cible = st.selectbox("Unité cible", ["mM", "M", "mg/mL"])
            
        if st.button("Calculer volume solvant"):
            if masse > 0 and conc_cible > 0:
                masse_g = masse / 1000.0
                vol_L = 0
                
                if unit_cible == "mg/mL":
                    conc_g_L = conc_cible 
                    vol_L = masse_g / conc_g_L
                else:
                    conc_M = get_base_value(conc_cible, unit_cible, "conc_molar")
                    if mw > 0: vol_L = masse_g / (mw * conc_M)
                    else: st.stop()
                
                # Affichage dynamique
                val_disp = 0
                unit_disp = ""
                if vol_L < 1e-3: 
                    val_disp, unit_disp = vol_L * 1e6, "µL"
                elif vol_L < 1: 
                    val_disp, unit_disp = vol_L * 1e3, "mL"
                else: 
                    val_disp, unit_disp = vol_L, "L"
                
                st.metric("Volume de solvant", f"{val_disp:.2f} {unit_disp}")
                st.info(f"📋 **Protocole :** Peser **{masse} mg** et dissoudre dans **{val_disp:.2f} {unit_disp}**.")
            else:
                st.error("Valeurs > 0 requises")

    # ============================================================================
    # 4. CONVERSION D'UNITÉS (MIS À JOUR AVEC OD et ENZYMES)
    # ============================================================================
    elif calculator_type == "🔄 Conversion d'unités (Avancé)":
        st.subheader("🔄 Convertisseur Bio/Chimie")
        
        # Sous-menu pour choisir le type de conversion
        conv_category = st.selectbox(
            "Type de conversion",
            ["Masse / Volume / Concentration", "Densité Optique (OD) ↔ Concentration", "Enzymes (Unités ↔ Masse)"]
        )
        
        st.markdown("---")

        if conv_category == "Masse / Volume / Concentration":
            type_conv = st.radio("Grandeur", ["Masse", "Volume", "Concentration"])
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

        # --- NOUVEAU MODULE OD ---
        elif conv_category == "Densité Optique (OD) ↔ Concentration":
            st.markdown("Conversion basée sur la loi de Beer-Lambert standard pour une cuve de 1 cm.")
            
            col1, col2 = st.columns(2)
            with col1:
                molecule = st.selectbox("Molécule", ["ADN Double Brin (dsDNA)", "ADN Simple Brin (ssDNA)", "ARN (RNA)", "Protéine (Personnalisé)"])
                od_val = st.number_input("Valeur OD (A260 ou A280)", value=1.0, min_value=0.0)
            
            with col2:
                # Facteurs standard (µg/mL pour 1 OD)
                factor = 0
                if "dsDNA" in molecule: 
                    factor = 50.0
                    st.info("ℹ️ Standard : 1 A260 = 50 µg/mL")
                elif "ssDNA" in molecule: 
                    factor = 33.0
                    st.info("ℹ️ Standard : 1 A260 = 33 µg/mL")
                elif "RNA" in molecule: 
                    factor = 40.0
                    st.info("ℹ️ Standard : 1 A260 = 40 µg/mL")
                else:
                    factor = st.number_input("Facteur de conversion (µg/mL pour 1 OD)", value=1000.0, help="Dépend du coefficient d'extinction de la protéine.")
            
            if st.button("Convertir en Concentration"):
                conc_ug_ml = od_val * factor
                st.success("✅ Concentration estimée :")
                c_res1, c_res2 = st.columns(2)
                with c_res1: st.metric("Concentration", f"{conc_ug_ml:.2f} µg/mL")
                with c_res2: st.metric("Concentration", f"{conc_ug_ml/1000:.3f} mg/mL")

        # --- NOUVEAU MODULE ENZYMES ---
        elif conv_category == "Enzymes (Unités ↔ Masse)":
            st.warning("⚠️ Attention : La conversion dépend de l'Activité Spécifique indiquée sur votre lot (CoA).")
            
            direction = st.radio("Sens du calcul", ["J'ai des Unités (U) → Je veux la Masse (mg)", "J'ai la Masse (mg) → Je veux les Unités (U)"])
            
            col1, col2 = st.columns(2)
            with col1:
                specific_activity = st.number_input("Activité Spécifique (U/mg)", min_value=1.0, value=2000.0, help="Trouvez cette valeur sur le certificat d'analyse du produit.")
            
            with col2:
                if "Unités (U) →" in direction:
                    units = st.number_input("Unités Totales (U)", value=1000.0)
                    if st.button("Calculer Masse"):
                        mass_mg = units / specific_activity
                        st.success(f"✅ Masse correspondante : **{mass_mg:.4f} mg** ({mass_mg*1000:.1f} µg)")
                else:
                    mass_input = st.number_input("Masse (mg)", value=1.0)
                    if st.button("Calculer Unités"):
                        units_res = mass_input * specific_activity
                        st.success(f"✅ Activité totale : **{units_res:.1f} U**")

    # ============================================================================
    # 5. pH ET TAMPONS
    # ============================================================================
    elif calculator_type == "🧪 pH et tampons":
        st.subheader("🧪 Calculateur de pH")
        ph_mode = st.selectbox("Type", ["Acide Fort", "Base Forte", "Acide Faible", "Base Faible", "Tampon"])
        col1, col2 = st.columns(2)
        
        if ph_mode == "Acide Fort":
            with col1: c = st.number_input("Concentration (M)", value=0.01, format="%.4f")
            if c > 0: st.metric("pH", f"{-math.log10(c):.2f}")
        elif ph_mode == "Base Forte":
            with col1: c = st.number_input("Concentration (M)", value=0.01, format="%.4f")
            if c > 0: st.metric("pH", f"{14 + math.log10(c):.2f}")
        elif ph_mode == "Acide Faible":
            with col1: c = st.number_input("Conc. (M)", value=0.1)
            with col2: pka = st.number_input("pKa", value=4.76)
            if c > 0: st.metric("pH", f"{0.5 * (pka - math.log10(c)):.2f}")
        elif ph_mode == "Base Faible":
            with col1: c = st.number_input("Conc. (M)", value=0.1)
            with col2: pka = st.number_input("pKa", value=9.25)
            if c > 0: st.metric("pH", f"{7 + 0.5 * (pka + math.log10(c)):.2f}")
        elif ph_mode == "Tampon":
            with col1: pka = st.number_input("pKa", value=4.76)
            with col2: ratio = st.number_input("Ratio [Base]/[Acide]", value=1.0)
            if ratio > 0: st.metric("pH", f"{pka + math.log10(ratio):.2f}")

    # ============================================================================
    # 6. DILUTIONS SÉRIE
    # ============================================================================
    elif calculator_type == "📈 Facteur de dilution série":
        st.subheader("📈 Gamme de Calibration")
        start = st.number_input("Conc. Départ", value=100.0)
        factor = st.number_input("Facteur", value=2.0)
        n = st.number_input("Nombre de points", value=8, step=1)
        if st.button("Générer"):
            df = pd.DataFrame([{"Point": i+1, "Concentration": start/(factor**i)} for i in range(int(n))])
            st.dataframe(df)

if __name__ == "__main__":
    app()
