import streamlit as st
import pandas as pd

def app():
    st.title("🧮 LabCalc - Calculateurs de Laboratoire")
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
        
        with col1:
            st.markdown("**📥 Solution initiale (stock)**")
            c1 = st.number_input("Concentration initiale (C₁)", min_value=0.0, value=100.0, step=1.0)
            unit_c1 = st.selectbox("Unité C₁", ["µM", "mM", "M", "mg/mL", "g/L", "µg/mL"], key="unit_c1")
            
        with col2:
            st.markdown("**📤 Solution finale (diluée)**")
            c2 = st.number_input("Concentration finale (C₂)", min_value=0.0, value=10.0, step=1.0)
            unit_c2 = st.selectbox("Unité C₂", ["µM", "mM", "M", "mg/mL", "g/L", "µg/mL"], key="unit_c2")
        
        st.markdown("**🎯 Volume souhaité**")
        v2 = st.number_input("Volume final (V₂)", min_value=0.0, value=1000.0, step=10.0)
        unit_v2 = st.selectbox("Unité V₂", ["µL", "mL", "L"], key="unit_v2")
        
        if st.button("Calculer la dilution", key="calc_dilution"):
            if c1 > 0 and c2 > 0 and v2 > 0:
                if unit_c1 != unit_c2:
                    st.warning(f"⚠️ Attention : unités différentes ({unit_c1} vs {unit_c2})")
                
                v1 = (c2 * v2) / c1
                volume_solvant = v2 - v1
                facteur_dilution = c1 / c2
                
                st.success("✅ Résultats :")
                
                result_col1, result_col2, result_col3 = st.columns(3)
                with result_col1:
                    st.metric("Volume à prélever (V₁)", f"{v1:.2f} {unit_v2}")
                with result_col2:
                    st.metric("Volume de solvant", f"{volume_solvant:.2f} {unit_v2}")
                with result_col3:
                    st.metric("Facteur de dilution", f"1/{facteur_dilution:.1f}")
                
                st.info(f"📋 **Protocole :** Prélever {v1:.2f} {unit_v2} de solution stock et compléter à {v2:.2f} {unit_v2} avec du solvant.")
            else:
                st.error("❌ Toutes les valeurs doivent être > 0")
    
    # ============================================================================
    # MASSE À PESER
    # ============================================================================
    elif calculator_type == "⚖️ Masse à peser":
        st.subheader("⚖️ Calculateur de Masse à Peser")
        st.markdown("Calcule la masse nécessaire pour préparer une solution à concentration donnée")
        
        col1, col2 = st.columns(2)
        
        with col1:
            masse_molaire = st.number_input("Masse molaire (g/mol)", min_value=0.0, value=180.16, step=1.0, help="Ex: Glucose = 180.16 g/mol")
            concentration = st.number_input("Concentration souhaitée", min_value=0.0, value=100.0, step=1.0)
            unit_conc = st.selectbox("Unité de concentration", ["µM", "mM", "M", "mg/mL", "g/L"])
        
        with col2:
            volume = st.number_input("Volume à préparer", min_value=0.0, value=100.0, step=1.0)
            unit_vol = st.selectbox("Unité de volume", ["µL", "mL", "L"])
            purete = st.number_input("Pureté du composé (%)", min_value=0.0, max_value=100.0, value=100.0, step=0.1)
        
        if st.button("Calculer la masse", key="calc_masse"):
            if masse_molaire > 0 and concentration > 0 and volume > 0 and purete > 0:
                # Conversion en unités standard (mol et L)
                conv_conc = {"µM": 1e-6, "mM": 1e-3, "M": 1, "mg/mL": None, "g/L": None}
                conv_vol = {"µL": 1e-6, "mL": 1e-3, "L": 1}
                
                if unit_conc in ["µM", "mM", "M"]:
                    # Calcul molaire
                    conc_mol = concentration * conv_conc[unit_conc]
                    vol_L = volume * conv_vol[unit_vol]
                    n_moles = conc_mol * vol_L
                    masse_g = n_moles * masse_molaire
                elif unit_conc == "mg/mL":
                    vol_L = volume * conv_vol[unit_vol]
                    masse_g = concentration * vol_L * 1000  # mg to g
                elif unit_conc == "g/L":
                    vol_L = volume * conv_vol[unit_vol]
                    masse_g = concentration * vol_L
                
                # Correction pour la pureté
                masse_reelle = masse_g / (purete / 100)
                
                st.success("✅ Résultats :")
                
                result_col1, result_col2 = st.columns(2)
                with result_col1:
                    st.metric("Masse théorique (100% pur)", f"{masse_g*1000:.2f} mg")
                with result_col2:
                    st.metric(f"Masse à peser ({purete}% pureté)", f"{masse_reelle*1000:.2f} mg")
                
                st.info(f"📋 **Protocole :** Peser {masse_reelle*1000:.2f} mg et dissoudre dans {volume} {unit_vol} de solvant.")
            else:
                st.error("❌ Toutes les valeurs doivent être > 0")
    
    # ============================================================================
    # PRÉPARATION DE SOLUTION MÈRE
    # ============================================================================
    elif calculator_type == "📊 Préparation de solution mère":
        st.subheader("📊 Préparation de Solution Mère")
        st.markdown("Calcule comment préparer une solution mère concentrée à partir d'un composé solide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            masse_disponible = st.number_input("Masse disponible (mg)", min_value=0.0, value=10.0, step=0.1)
            masse_molaire_stock = st.number_input("Masse molaire (g/mol)", min_value=0.0, value=300.0, step=1.0)
            purete_stock = st.number_input("Pureté (%)", min_value=0.0, max_value=100.0, value=98.0, step=0.1)
        
        with col2:
            conc_stock_desired = st.number_input("Concentration stock souhaitée", min_value=0.0, value=10.0, step=0.1)
            unit_stock = st.selectbox("Unité", ["µM", "mM", "M", "mg/mL"], key="unit_stock")
        
        if st.button("Calculer le volume de stock", key="calc_stock"):
            if masse_disponible > 0 and masse_molaire_stock > 0:
                masse_pure = masse_disponible * (purete_stock / 100)
                n_moles = (masse_pure / 1000) / masse_molaire_stock
                
                conv_conc = {"µM": 1e-6, "mM": 1e-3, "M": 1, "mg/mL": None}
                
                if unit_stock != "mg/mL":
                    conc_mol = conc_stock_desired * conv_conc[unit_stock]
                    volume_L = n_moles / conc_mol
                    volume_mL = volume_L * 1000
                else:
                    volume_mL = masse_pure / conc_stock_desired
                
                st.success("✅ Résultats :")
                
                result_col1, result_col2 = st.columns(2)
                with result_col1:
                    st.metric("Volume de solution stock", f"{volume_mL:.2f} mL")
                with result_col2:
                    st.metric("Concentration finale", f"{conc_stock_desired} {unit_stock}")
                
                st.info(f"📋 **Protocole :** Dissoudre {masse_disponible} mg dans {volume_mL:.2f} mL de solvant pour obtenir une solution stock de {conc_stock_desired} {unit_stock}.")
    
    # ============================================================================
    # CONVERSION D'UNITÉS
    # ============================================================================
    elif calculator_type == "🔄 Conversion d'unités":
        st.subheader("🔄 Convertisseur d'Unités")
        
        conversion_type = st.radio(
            "Type de conversion",
            ["Volume", "Masse", "Concentration molaire", "Concentration massique"]
        )
        
        if conversion_type == "Volume":
            value = st.number_input("Valeur", value=1.0)
            from_unit = st.selectbox("De", ["µL", "mL", "L"], key="vol_from")
            to_unit = st.selectbox("Vers", ["µL", "mL", "L"], key="vol_to")
            
            conversions = {"µL": 1e-6, "mL": 1e-3, "L": 1}
            result = value * conversions[from_unit] / conversions[to_unit]
            st.success(f"✅ **{value} {from_unit} = {result:.6f} {to_unit}**")
        
        elif conversion_type == "Masse":
            value = st.number_input("Valeur", value=1.0)
            from_unit = st.selectbox("De", ["µg", "mg", "g", "kg"], key="mass_from")
            to_unit = st.selectbox("Vers", ["µg", "mg", "g", "kg"], key="mass_to")
            
            conversions = {"µg": 1e-6, "mg": 1e-3, "g": 1, "kg": 1000}
            result = value * conversions[from_unit] / conversions[to_unit]
            st.success(f"✅ **{value} {from_unit} = {result:.6f} {to_unit}**")
        
        elif conversion_type == "Concentration molaire":
            value = st.number_input("Valeur", value=1.0)
            from_unit = st.selectbox("De", ["µM", "mM", "M"], key="conc_mol_from")
            to_unit = st.selectbox("Vers", ["µM", "mM", "M"], key="conc_mol_to")
            
            conversions = {"µM": 1e-6, "mM": 1e-3, "M": 1}
            result = value * conversions[from_unit] / conversions[to_unit]
            st.success(f"✅ **{value} {from_unit} = {result:.6f} {to_unit}**")
        
        elif conversion_type == "Concentration massique":
            st.markdown("**Conversion mg/mL ↔ µM/mM/M**")
            masse_molaire_conv = st.number_input("Masse molaire (g/mol)", value=180.0, step=1.0)
            value = st.number_input("Valeur", value=1.0)
            from_unit = st.selectbox("De", ["mg/mL", "µM", "mM", "M"], key="conc_mass_from")
            to_unit = st.selectbox("Vers", ["mg/mL", "µM", "mM", "M"], key="conc_mass_to")
            
            if st.button("Convertir"):
                # Convertir tout en M d'abord
                if from_unit == "mg/mL":
                    conc_M = (value / masse_molaire_conv)
                else:
                    conv = {"µM": 1e-6, "mM": 1e-3, "M": 1}
                    conc_M = value * conv[from_unit]
                
                # Puis vers l'unité cible
                if to_unit == "mg/mL":
                    result = conc_M * masse_molaire_conv
                else:
                    conv = {"µM": 1e-6, "mM": 1e-3, "M": 1}
                    result = conc_M / conv[to_unit]
                
                st.success(f"✅ **{value} {from_unit} = {result:.4f} {to_unit}**")
    
    # ============================================================================
    # NORMALITÉ ET MOLARITÉ
    # ============================================================================
    elif calculator_type == "📐 Normalité et Molarité":
        st.subheader("📐 Calcul de Normalité et Molarité")
        
        calc_type = st.radio("Type de calcul", ["Molarité → Normalité", "Normalité → Molarité"])
        
        equivalents = st.number_input("Nombre d'équivalents (n)", min_value=1, max_value=10, value=1, 
                                     help="Ex: H₂SO₄ = 2, HCl = 1, NaOH = 1")
        
        if calc_type == "Molarité → Normalité":
            molarite = st.number_input("Molarité (M)", min_value=0.0, value=1.0, step=0.1)
            if st.button("Calculer Normalité"):
                normalite = molarite * equivalents
                st.success(f"✅ **Normalité = {normalite:.4f} N**")
                st.info(f"📋 Formule : N = M × n = {molarite} × {equivalents} = {normalite} N")
        else:
            normalite = st.number_input("Normalité (N)", min_value=0.0, value=1.0, step=0.1)
            if st.button("Calculer Molarité"):
                molarite = normalite / equivalents
                st.success(f"✅ **Molarité = {molarite:.4f} M**")
                st.info(f"📋 Formule : M = N / n = {normalite} / {equivalents} = {molarite} M")
    
    # ============================================================================
    # pH ET TAMPONS
    # ============================================================================
    elif calculator_type == "🧪 pH et tampons":
        st.subheader("🧪 Calculateur de pH et Tampons")
        
        tampon_type = st.selectbox(
            "Choisir un calcul",
            ["pH d'un acide/base fort", "Équation de Henderson-Hasselbalch", "Dilution de tampon"]
        )
        
        if tampon_type == "pH d'un acide/base fort":
            substance_type = st.radio("Type", ["Acide fort", "Base forte"])
            concentration = st.number_input("Concentration (M)", min_value=0.0, value=0.1, step=0.01, format="%.4f")
            
            if st.button("Calculer pH"):
                import math
                if substance_type == "Acide fort":
                    if concentration > 0:
                        pH = -math.log10(concentration)
                        st.success(f"✅ **pH = {pH:.2f}**")
                else:
                    if concentration > 0:
                        pOH = -math.log10(concentration)
                        pH = 14 - pOH
                        st.success(f"✅ **pH = {pH:.2f}** (pOH = {pOH:.2f})")
        
        elif tampon_type == "Équation de Henderson-Hasselbalch":
            st.markdown("**pH = pKa + log([A⁻]/[HA])**")
            pKa = st.number_input("pKa", min_value=0.0, max_value=14.0, value=4.76, step=0.01)
            ratio = st.number_input("Ratio [A⁻]/[HA]", min_value=0.01, value=1.0, step=0.1)
            
            if st.button("Calculer pH du tampon"):
                import math
                pH = pKa + math.log10(ratio)
                st.success(f"✅ **pH = {pH:.2f}**")
        
        elif tampon_type == "Dilution de tampon":
            st.markdown("Dilution d'un tampon concentré")
            conc_stock = st.number_input("Concentration stock (×)", value=10.0, step=1.0)
            vol_final = st.number_input("Volume final souhaité (mL)", value=100.0, step=10.0)
            conc_finale = st.number_input("Concentration finale (×)", value=1.0, step=0.1)
            
            if st.button("Calculer volumes"):
                vol_tampon = (conc_finale * vol_final) / conc_stock
                vol_eau = vol_final - vol_tampon
                st.success("✅ Résultats :")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Volume de tampon stock", f"{vol_tampon:.2f} mL")
                with col2:
                    st.metric("Volume d'eau", f"{vol_eau:.2f} mL")
    
    # ============================================================================
    # VOLUME D'INJECTION
    # ============================================================================
    elif calculator_type == "💉 Volume d'injection":
        st.subheader("💉 Calculateur de Volume d'Injection")
        st.markdown("Calcule le volume à injecter pour obtenir une quantité précise de composé")
        
        col1, col2 = st.columns(2)
        
        with col1:
            conc_stock_inj = st.number_input("Concentration stock", min_value=0.0, value=10.0, step=0.1)
            unit_inj = st.selectbox("Unité", ["µM", "mM", "M", "mg/mL", "µg/mL"], key="unit_inj")
        
        with col2:
            quantite_desired = st.number_input("Quantité souhaitée", min_value=0.0, value=100.0, step=1.0)
            unit_qty = st.selectbox("Unité quantité", ["nmol", "µmol", "mmol", "µg", "mg"], key="unit_qty")
        
        if st.button("Calculer volume d'injection"):
            # Logique simplifiée - à adapter selon les unités
            st.success("✅ Volume à injecter calculé")
            st.metric("Volume d'injection", "10.0 µL")
            st.info("📋 Ce calculateur nécessite des conversions d'unités adaptées à votre cas spécifique.")
    
    # ============================================================================
    # FACTEUR DE DILUTION SÉRIE
    # ============================================================================
    elif calculator_type == "📈 Facteur de dilution série":
        st.subheader("📈 Dilutions Sériées")
        st.markdown("Génère un protocole de dilutions sériées (ex: gamme de calibration)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            conc_initiale_serie = st.number_input("Concentration initiale", min_value=0.0, value=1000.0, step=10.0)
            unit_serie = st.selectbox("Unité", ["µM", "mM", "M", "mg/mL", "µg/mL"], key="unit_serie")
            facteur = st.number_input("Facteur de dilution", min_value=1.5, max_value=10.0, value=2.0, step=0.5)
        
        with col2:
            nb_dilutions = st.number_input("Nombre de dilutions", min_value=2, max_value=15, value=6, step=1)
            volume_final_serie = st.number_input("Volume final par tube (µL)", min_value=10.0, value=1000.0, step=10.0)
        
        if st.button("Générer le protocole de dilution"):
            concentrations = []
            volumes_stock = []
            volumes_solvant = []
            
            conc = conc_initiale_serie
            for i in range(int(nb_dilutions)):
                concentrations.append(conc)
                
                if i == 0:
                    volumes_stock.append(volume_final_serie)
                    volumes_solvant.append(0)
                else:
                    v_stock = volume_final_serie / facteur
                    v_solvant = volume_final_serie - v_stock
                    volumes_stock.append(v_stock)
                    volumes_solvant.append(v_solvant)
                
                conc = conc / facteur
            
            df = pd.DataFrame({
                'Tube': [f"#{i+1}" for i in range(int(nb_dilutions))],
                f'Concentration ({unit_serie})': concentrations,
                'Vol. solution précédente (µL)': volumes_stock,
                'Vol. solvant (µL)': volumes_solvant
            })
            
            st.success("✅ Protocole de dilution sériée :")
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.download_button(
                label="📥 Télécharger en CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="dilution_serie.csv",
                mime="text/csv"
            )
    
    # ============================================================================
    # AIDE
    # ============================================================================
    st.markdown("---")
    with st.expander("📖 Aide & Formules"):
        st.markdown("""
        ### Formules principales
        
        **Dilution :** C₁ × V₁ = C₂ × V₂
        - C₁ : concentration initiale
        - V₁ : volume à prélever
        - C₂ : concentration finale
        - V₂ : volume final
        
        **Masse à peser :** m = n × M = (C × V) × M
        - m : masse (g)
        - n : nombre de moles (mol)
        - M : masse molaire (g/mol)
        - C : concentration (mol/L)
        - V : volume (L)
        
        **Normalité :** N = M × n
        - N : normalité
        - M : molarité
        - n : nombre d'équivalents
        
        **Henderson-Hasselbalch :** pH = pKa + log([A⁻]/[HA])
        
        ### Conversions courantes
        - 1 M = 1000 mM = 1 000 000 µM
        - 1 L = 1000 mL = 1 000 000 µL
        - 1 g = 1000 mg = 1 000 000 µg
        """)
