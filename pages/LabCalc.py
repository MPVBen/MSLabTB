import streamlit as st
import pandas as pd
import math

# Dictionnaires de conversion vers les unités de base (Molaire, Litre, Gramme)
CONV_CONC_MOLAR = {"µM": 1e-6, "mM": 1e-3, "M": 1.0}
CONV_VOL = {"nL": 1e-9, "µL": 1e-6, "mL": 1e-3, "L": 1.0}
CONV_MASS = {"ng": 1e-9, "µg": 1e-6, "mg": 1e-3, "g": 1.0, "kg": 1000.0}

def get_base_value(value, unit, type_measure="conc_molar"):
    """Convertit une valeur vers son unité SI de base."""
    if type_measure == "conc_molar" and unit in CONV_CONC_MOLAR:
        return value * CONV_CONC_MOLAR[unit]
    elif type_measure == "vol" and unit in CONV_VOL:
        return value * CONV_VOL[unit]
    elif type_measure == "mass" and unit in CONV_MASS:
        return value * CONV_MASS[unit]
    # Cas spécifiques pour concentration massique (base = g/L)
    elif type_measure == "conc_mass":
        if unit == "g/L" or unit == "mg/mL": return value
        if unit == "µg/mL": return value * 1e-3
        if unit == "ng/µL": return value * 1e-3
    return None

def app():
    st.set_page_config(page_title="LabCalc", page_icon="🧮")
    st.title("🧮 LabCalc - Calculateurs de Laboratoire (Corrigé)")
    st.markdown("---")
    
    # Menu de sélection des calculateurs
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
    # CALCULATEUR DE DILUTION (C1V1 = C2V2)
    # ============================================================================
    if calculator_type == "💧 Dilution (C1V1 = C2V2)":
        st.subheader("💧 Calculateur de Dilution")
        st.markdown("Formule : **C₁ × V₁ = C₂ × V₂**")
        
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
                # Vérification de compatibilité des unités
                is_c1_molar = unit_c1 in units_molar
                is_c2_molar = unit_c2 in units_molar
                
                if is_c1_molar != is_c2_molar:
                    st.error("⛔ Erreur : Impossible de diluer une concentration Molaire vers Massique (ou inversement) sans la Masse Molaire. Utilisez le même type d'unité.")
                else:
                    # Normalisation des valeurs pour le calcul
                    type_c = "conc_molar" if is_c1_molar else "conc_mass"
                    c1_norm = get_base_value(c1, unit_c1, type_c)
                    c2_norm = get_base_value(c2, unit_c2, type_c)
                    
                    if c2_norm >= c1_norm:
                        st.error("⛔ Erreur : La concentration finale (C2) doit être inférieure à la concentration stock (C1).")
                    else:
                        # Calcul
                        # V1 = (C2 * V2) / C1 -> Les unités de volume s'annulent ou se conservent, 
                        # on peut garder V2 dans son unité d'origine pour que V1 sorte dans la même unité.
                        # Seules les concentrations doivent être normalisées pour avoir le bon ratio.
                        
                        v1 = (c2_norm * v2) / c1_norm
                        volume_solvant = v2 - v1
                        facteur_dilution = c1_norm / c2_norm
                        
                        st.success("✅ Résultats :")
                        
                        result_col1, result_col2, result_col3 = st.columns(3)
                        with result_col1:
                            st.metric("Volume à prélever (V₁)", f"{v1:.4f} {unit_v2}")
                        with result_col2:
                            st.metric("Volume de solvant", f"{volume_solvant:.4f} {unit_v2}")
                        with result_col3:
                            st.metric("Facteur de dilution", f"1 : {facteur_dilution:.1f}")
                        
                        st.info(f"📋 **Protocole :** Prélever **{v1:.4f} {unit_v2}** de solution stock et compléter avec **{volume_solvant:.4f} {unit_v2}** de solvant.")
            else:
                st.error("❌ Toutes les valeurs doivent être > 0")
    
    # ============================================================================
    # MASSE À PESER
    # ============================================================================
    elif calculator_type == "⚖️ Masse à peser":
        st.subheader("⚖️ Calculateur de Masse à Peser")
        
        col1, col2 = st.columns(2)
        
        with col1:
            masse_molaire = st.number_input("Masse molaire (g/mol)", min_value=0.0, value=180.16, step=1.0)
            concentration = st.number_input("Concentration souhaitée", min_value=0.0, value=100.0, step=1.0)
            unit_conc = st.selectbox("Unité de concentration", ["µM", "mM", "M", "mg/mL", "g/L"])
        
        with col2:
            volume = st.number_input("Volume à préparer", min_value=0.0, value=100.0, step=1.0)
            unit_vol = st.selectbox("Unité de volume", ["µL", "mL", "L"])
            purete = st.number_input("Pureté du composé (%)", min_value=0.0, max_value=100.0, value=100.0, step=0.1)
        
        if st.button("Calculer la masse", key="calc_masse"):
            if concentration > 0 and volume > 0:
                # Normalisation Volume en Litres
                vol_L = get_base_value(volume, unit_vol, "vol")
                masse_g = 0
                
                # Calcul selon le type d'unité
                if unit_conc in ["µM", "mM", "M"]:
                    if masse_molaire <= 0:
                        st.error("Masse molaire requise pour ce calcul.")
                        return
                    conc_mol_L = get_base_value(concentration, unit_conc, "conc_molar")
                    n_moles = conc_mol_L * vol_L
                    masse_g = n_moles * masse_molaire
                else:
                    # Unités massiques (g/L, mg/mL)
                    # Note : 1 mg/mL = 1 g/L
                    conc_g_L = get_base_value(concentration, unit_conc, "conc_mass")
                    masse_g = conc_g_L * vol_L
                
                # Correction pureté
                masse_reelle_g = masse_g / (purete / 100)
                
                # Conversion intelligente pour l'affichage (si < 1mg -> µg, si < 1g -> mg)
                if masse_reelle_g < 1e-3:
                     display_val = masse_reelle_g * 1e6
                     display_unit = "µg"
                elif masse_reelle_g < 1:
                     display_val = masse_reelle_g * 1e3
                     display_unit = "mg"
                else:
                     display_val = masse_reelle_g
                     display_unit = "g"
                
                st.success("✅ Résultats :")
                st.metric(f"Masse à peser ({purete}% pureté)", f"{display_val:.4f} {display_unit}")
                
                if purete < 100:
                    masse_pure_disp = (masse_g * 1e3) if masse_g < 1 else masse_g
                    unit_pure = "mg" if masse_g < 1 else "g"
                    st.caption(f"(Correspond à {masse_pure_disp:.4f} {unit_pure} de produit pur)")

    # ============================================================================
    # PRÉPARATION DE SOLUTION MÈRE
    # ============================================================================
    elif calculator_type == "📊 Préparation de solution mère":
        st.subheader("📊 Préparation de Solution Mère")
        
        col1, col2 = st.columns(2)
        with col1:
            masse_disponible = st.number_input("Masse disponible (mg)", min_value=0.0, value=10.0, step=0.1)
            masse_molaire_stock = st.number_input("Masse molaire (g/mol)", min_value=0.0, value=300.0, step=1.0)
            purete_stock = st.number_input("Pureté (%)", min_value=0.0, max_value=100.0, value=98.0, step=0.1)
        with col2:
            conc_stock_desired = st.number_input("Concentration stock souhaitée", min_value=0.0, value=10.0, step=0.1)
            unit_stock = st.selectbox("Unité", ["µM", "mM", "M", "mg/mL"], key="unit_stock")
        
        if st.button("Calculer le volume", key="calc_stock"):
            if masse_disponible > 0:
                masse_pure_g = (masse_disponible * (purete_stock / 100)) / 1000.0
                volume_L = 0
                
                if unit_stock in ["µM", "mM", "M"]:
                    if masse_molaire_stock <= 0:
                         st.error("Masse molaire requise.")
                         return
                    n_moles = masse_pure_g / masse_molaire_stock
                    conc_M = get_base_value(conc_stock_desired, unit_stock, "conc_molar")
                    volume_L = n_moles / conc_M
                else:
                    # mg/mL = g/L
                    conc_g_L = get_base_value(conc_stock_desired, unit_stock, "conc_mass")
                    volume_L = masse_pure_g / conc_g_L
                
                # Affichage intelligent du volume
                if volume_L < 1e-3:
                    st.metric("Volume de solvant à ajouter", f"{volume_L*1e6:.2f} µL")
                    unit_disp = "µL"
                    val_disp = volume_L*1e6
                else:
                    st.metric("Volume de solvant à ajouter", f"{volume_L*1e3:.2f} mL")
                    unit_disp = "mL"
                    val_disp = volume_L*1e3
                
                st.info(f"📋 Dissoudre toute la poudre dans **{val_disp:.2f} {unit_disp}** de solvant.")

    # ============================================================================
    # CONVERSION D'UNITÉS
    # ============================================================================
    elif calculator_type == "🔄 Conversion d'unités":
        st.subheader("🔄 Convertisseur d'Unités")
        conversion_type = st.radio("Type de conversion", ["Volume", "Masse", "Concentration molaire", "Concentration massique"])
        
        col1, col2 = st.columns(2)
        if conversion_type == "Volume":
            with col1:
                val = st.number_input("Valeur", value=1.0)
                u_from = st.selectbox("De", list(CONV_VOL.keys()))
            with col2:
                u_to = st.selectbox("Vers", list(CONV_VOL.keys()))
            res = (val * CONV_VOL[u_from]) / CONV_VOL[u_to]
            st.success(f"**{val} {u_from} = {res:.6g} {u_to}**")
            
        elif conversion_type == "Masse":
            with col1:
                val = st.number_input("Valeur", value=1.0)
                u_from = st.selectbox("De", list(CONV_MASS.keys()))
            with col2:
                u_to = st.selectbox("Vers", list(CONV_MASS.keys()))
            res = (val * CONV_MASS[u_from]) / CONV_MASS[u_to]
            st.success(f"**{val} {u_from} = {res:.6g} {u_to}**")
            
        elif conversion_type == "Concentration molaire":
            with col1:
                val = st.number_input("Valeur", value=1.0)
                u_from = st.selectbox("De", list(CONV_CONC_MOLAR.keys()))
            with col2:
                u_to = st.selectbox("Vers", list(CONV_CONC_MOLAR.keys()))
            res = (val * CONV_CONC_MOLAR[u_from]) / CONV_CONC_MOLAR[u_to]
            st.success(f"**{val} {u_from} = {res:.6g} {u_to}**")
            
        elif conversion_type == "Concentration massique":
            st.info("Conversion mg/mL ↔ Molaire (Nécessite MW)")
            mw = st.number_input("Masse molaire (g/mol)", value=180.0)
            c1, c2 = st.columns(2)
            with c1:
                val = st.number_input("Valeur", value=1.0)
                u_from = st.selectbox("De", ["mg/mL", "M", "mM", "µM"], key="c_mass_from")
            with c2:
                u_to = st.selectbox("Vers", ["mg/mL", "M", "mM", "µM"], key="c_mass_to")
                
            if st.button("Convertir"):
                # Etape 1 : Tout convertir en Molaire (mol/L)
                val_M = 0
                if u_from == "mg/mL":
                    # mg/mL = g/L -> / MW -> mol/L
                    val_M = val / mw
                else:
                    val_M = val * CONV_CONC_MOLAR[u_from]
                
                # Etape 2 : Molaire vers Cible
                res = 0
                if u_to == "mg/mL":
                    res = val_M * mw
                else:
                    res = val_M / CONV_CONC_MOLAR[u_to]
                    
                st.success(f"**{val} {u_from} = {res:.4f} {u_to}**")

    # ============================================================================
    # NORMALITÉ ET MOLARITÉ
    # ============================================================================
    elif calculator_type == "📐 Normalité et Molarité":
        st.subheader("📐 Calcul de Normalité et Molarité")
        col1, col2 = st.columns(2)
        with col1:
            calc_type = st.radio("Type", ["Molarité → Normalité", "Normalité → Molarité"])
        with col2:
            equivalents = st.number_input("Équivalents (n)", min_value=1, value=1, help="Ex: H2SO4 = 2")
            
        if calc_type == "Molarité → Normalité":
            val = st.number_input("Molarité (M)", value=1.0)
            if st.button("Calculer"):
                st.success(f"Normalité = {val * equivalents:.4f} N")
        else:
            val = st.number_input("Normalité (N)", value=1.0)
            if st.button("Calculer"):
                st.success(f"Molarité = {val / equivalents:.4f} M")

    # ============================================================================
    # pH ET TAMPONS
    # ============================================================================
    elif calculator_type == "🧪 pH et tampons":
        st.subheader("🧪 Calculateur de pH")
        tampon_type = st.selectbox("Mode", ["Acide fort / Base forte", "Henderson-Hasselbalch (Tampon)", "Dilution de tampon"])
        
        if tampon_type == "Acide fort / Base forte":
            c1, c2 = st.columns(2)
            with c1: type_sub = st.radio("Substance", ["Acide fort", "Base forte"])
            with c2: conc = st.number_input("Concentration (M)", format="%.4f", value=0.01)
            if st.button("Calculer pH"):
                if conc > 0:
                    pH = -math.log10(conc) if type_sub == "Acide fort" else 14 + math.log10(conc)
                    st.metric("pH", f"{pH:.2f}")
                else: st.error("Concentration > 0 requise")
                
        elif tampon_type == "Henderson-Hasselbalch (Tampon)":
            c1, c2 = st.columns(2)
            with c1: pKa = st.number_input("pKa", value=4.76)
            with c2: ratio = st.number_input("Ratio [Base]/[Acide]", value=1.0)
            if st.button("Calculer"):
                if ratio > 0:
                    pH = pKa + math.log10(ratio)
                    st.metric("pH", f"{pH:.2f}")
                else: st.error("Ratio doit être > 0")
        
        elif tampon_type == "Dilution de tampon":
            st.info("Utilisez le calculateur de Dilution principal (C1V1) pour plus d'options.")
            
    # ============================================================================
    # VOLUME D'INJECTION (CORRIGÉ ET IMPLÉMENTÉ)
    # ============================================================================
    elif calculator_type == "💉 Volume d'injection":
        st.subheader("💉 Calculateur de Volume d'Injection")
        st.markdown("Calcule le volume à injecter pour délivrer une masse ou une quantité de matière précise.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Solution Stock**")
            conc_val = st.number_input("Concentration", value=1.0, format="%.2f")
            conc_unit = st.selectbox("Unité", ["mg/mL", "µg/mL", "M", "mM", "µM"])
            
            if conc_unit in ["M", "mM", "µM"]:
                mw = st.number_input("Masse molaire (g/mol)", value=300.0, help="Nécessaire pour convertir moles en masse si besoin")
            else:
                mw = None

        with col2:
            st.markdown("**Quantité à injecter**")
            qty_val = st.number_input("Quantité", value=10.0)
            qty_unit = st.selectbox("Unité quantité", ["mg", "µg", "ng", "mmol", "µmol", "nmol"])
            
        if st.button("Calculer Volume d'injection"):
            # 1. Convertir la concentration stock en unité standard (g/L ou mol/L)
            c_base = 0
            is_molar_conc = conc_unit in ["M", "mM", "µM"]
            
            if is_molar_conc:
                c_base = get_base_value(conc_val, conc_unit, "conc_molar") # mol/L
            else:
                c_base = get_base_value(conc_val, conc_unit, "conc_mass") # g/L
            
            # 2. Convertir la quantité désirée en standard (g ou mol)
            q_base = 0
            is_molar_qty = qty_unit in ["mmol", "µmol", "nmol"]
            
            # Conversion manuelle simple pour moles car pas dans dict global
            conv_mol_qty = {"mmol": 1e-3, "µmol": 1e-6, "nmol": 1e-9}
            
            if is_molar_qty:
                q_base = qty_val * conv_mol_qty[qty_unit] # mol
            else:
                q_base = get_base_value(qty_val, qty_unit, "mass") # g
                
            # 3. Calcul du volume (V = n/C ou V = m/C)
            # Il faut gérer la compatibilité Moles/Masse
            vol_L = 0
            
            error = False
            if is_molar_conc and is_molar_qty:
                vol_L = q_base / c_base
            elif not is_molar_conc and not is_molar_qty:
                vol_L = q_base / c_base
            elif is_molar_conc and not is_molar_qty:
                # Stock en Molaire, on veut injecter des Grammes
                # C (mol/L) = m / (MW * V)  -> V = m / (MW * C)
                if mw > 0: vol_L = q_base / (mw * c_base)
                else: error = True
            elif not is_molar_conc and is_molar_qty:
                # Stock en g/L, on veut injecter des Moles
                # C (g/L) = (n * MW) / V -> V = (n * MW) / C
                if mw and mw > 0: vol_L = (q_base * mw) / c_base
                else: error = True
            
            if error:
                st.error("⚠️ Conversion impossible sans Masse Molaire (MW).")
            else:
                # Affichage adaptatif
                if vol_L < 1e-6:
                    st.metric("Volume à injecter", f"{vol_L*1e9:.2f} nL")
                elif vol_L < 1e-3:
                    st.metric("Volume à injecter", f"{vol_L*1e6:.2f} µL")
                else:
                    st.metric("Volume à injecter", f"{vol_L*1e3:.2f} mL")

    # ============================================================================
    # FACTEUR DE DILUTION SÉRIE
    # ============================================================================
    elif calculator_type == "📈 Facteur de dilution série":
        st.subheader("📈 Dilutions Sériées")
        c1, c2 = st.columns(2)
        with c1:
            conc = st.number_input("Conc. initiale", value=100.0)
            facteur = st.number_input("Facteur de dilution (X)", value=2.0, min_value=1.1)
        with c2:
            nb = st.number_input("Nombre de points", value=8, step=1)
            vol = st.number_input("Volume final par puit (µL)", value=100.0)
            
        if st.button("Générer table"):
            data = []
            current_conc = conc
            # Calcul du volume de transfert pour atteindre le facteur
            # Facteur F = V_total / V_transfert = (V_trans + V_solv) / V_trans
            # V_trans = V_total / F
            # Ici "Volume final par puit" est souvent le volume total de travail.
            # Donc pour préparer ça :
            v_transfert = vol / (facteur - 1) # Attention logique
            # Rectification logique standard : 
            # On veut V_final dans le puit.
            # Si on fait une dilution 1/2 (facteur 2). On met 100µL solvant, on ajoute 100µL stock. Total 200. On retire 100. Reste 100.
            # Simplifions : Affichons les concentrations.
            
            for i in range(int(nb)):
                data.append({"Puit": i+1, "Concentration": f"{current_conc:.4g}", "Dilution": f"1/{facteur**i:.1f}"})
                current_conc /= facteur
                
            st.dataframe(pd.DataFrame(data))
            st.caption(f"Pour un facteur {facteur}X : Mélanger 1 part de stock + {facteur-1} parts de solvant.")

if __name__ == "__main__":
    app()
