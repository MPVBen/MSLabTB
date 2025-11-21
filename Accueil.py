import streamlit as st
import os
import importlib

# Cacher le menu de navigation natif Streamlit
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

ICON_FOLDER = "assets"

# Correspondance nom logiciel / nom fichier python dans pages/ / icône SVG
apps = {
    "🏠 Accueil": {"module": None, "icon": "TB_logo.svg"},
    "MassCalc": {"module": "MassCalc", "icon": "icone_MC.svg"},
    "BDTool": {"module": "BDTool", "icon": "icone_BD.svg"},
    "KDTool": {"module": "KDTool", "icon": "icone_KD.svg"},
    "ThermoTool statistic fit": {"module": "ThermoTool_statistic_fit", "icon": "icone_TT.svg"},
    "ThermoTool gamma incomplete fit": {"module": "ThermoTool_gamma_incomplete_fit", "icon": "icone_TT.svg"},
    "LabCalc": {"module": "LabCalc", "icon": "icone_LC.svg"},
}

# Initialisation sur la page d'accueil par défaut
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Accueil"

st.sidebar.title("MS Lab Toolbox")
st.sidebar.markdown("---")

# Menu avec icônes
for app_name, info in apps.items():
    icon_path = os.path.join(ICON_FOLDER, info["icon"])
    
    col1, col2 = st.sidebar.columns([0.2, 0.8])
    
    with col1:
        if os.path.exists(icon_path):
            st.image(icon_path, width=38)
        else:
            st.write("❓")
    
    with col2:
        if st.button(app_name, key=f"btn_{app_name}", use_container_width=True):
            st.session_state.page = app_name

st.sidebar.markdown("---")

# Affichage du contenu selon la page sélectionnée
if st.session_state.page == "🏠 Accueil":
    # ============================================================================
    # PAGE D'ACCUEIL
    # ============================================================================
    st.image("assets/TB_logo.svg", width=200)
    st.title("🔬 MS Lab Toolbox")
    st.markdown("---")
    
    st.markdown("""
    ## Bienvenue dans la boîte à outils du laboratoire de spectrométrie de masse !
    
    Cette application regroupe plusieurs outils utiles pour l'analyse et le traitement de données MS.
    
    ### 📚 Outils disponibles :
    
    - **BDTool** : Visualisation et analyse de courbes de dissociation (breakdown curves)
    - **KDTool** : Détermination de constantes de dissociation par titrage MS (méthodes Hill & Gabelica)
    - **MassCalc** : Calculateur de masse moléculaire et m/z
    - **ThermoTool** : Calculs de température effective avec ajustements statistiques et gamma
    
    ### 🚀 Pour commencer :
    
    1. Sélectionnez un outil dans le menu de gauche
    2. Suivez les instructions spécifiques à chaque outil
    3. Téléchargez vos résultats en fin d'analyse
    """)
    
    # ============================================================================
    # SECTION TÉLÉCHARGEMENT DE TEMPLATES
    # ============================================================================
    st.markdown("---")
    st.subheader("📥 Templates Excel")
    st.markdown("Téléchargez les fichiers templates pour faciliter l'utilisation des outils :")
    
    # Création des colonnes pour les templates
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Template BDTool**")
        st.caption("Format pour les courbes de dissociation (breakdown curves)")
        
        # Lire le fichier template existant
        template_bd_path = "templates/Template_BD.xlsx"
        if os.path.exists(template_bd_path):
            with open(template_bd_path, "rb") as file:
                st.download_button(
                    label="⬇️ Télécharger Template_BD.xlsx",
                    data=file,
                    file_name="Template_BD.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("⚠️ Template non trouvé. Placez Template_BD.xlsx dans le dossier 'templates/'")
    
    with col2:
        st.markdown("**🧪 Template KDTool**")
        st.caption("Format pour la détermination de KD (titration)")
        
        # Lire le fichier template existant
        template_kd_path = "templates/Template_KD.xlsx"
        if os.path.exists(template_kd_path):
            with open(template_kd_path, "rb") as file:
                st.download_button(
                    label="⬇️ Télécharger Template_KD.xlsx",
                    data=file,
                    file_name="Template_KD.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("⚠️ Template non trouvé. Placez Template_KD.xlsx dans le dossier 'templates/'")
    
    st.markdown("---")
    
    # Suite de la page d'accueil
    st.markdown("""
    ### 📖 Instructions générales :
    
    - **BDTool** et **KDTool** nécessitent des fichiers au format spécifique (téléchargez les templates ci-dessus)
    - **BDTool** peut également lire les fichiers générés par TWIMExtract (https://sites.lsa.umich.edu/ruotolo/software/twim-extract/)
    - Les résultats peuvent être exportés en Excel, SVG ou PDF selon l'outil
    - Encodage recommandé : UTF-8
    - Pour toute question, contactez l'équipe du laboratoire
    
    ### ⚙️ Configuration :
    
    - **Formats supportés** : CSV, Excel
    - **Langues disponibles** : Français / English (selon l'outil)
    - **Instruments** : Compatibilité avec la plupart des spectromètres de masse
    
    ---
    
    💡 **Astuce** : Téléchargez les templates ci-dessus et remplissez-les avec vos données pour démarrer rapidement !
    """)
    
    # Section avec colonnes d'information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🧪 **Instruments**\n\nIM-MS, Orbitrap, Q-TOF, MALDI")
    
    with col2:
        st.success("📊 **Formats**\n\nCSV, Excel")
    
    with col3:
        st.warning("🔄 **Version**\n\n2.0 - Novembre 2025")
    
    # Section optionnelle : Changelog ou actualités
    with st.expander("📋 Dernières mises à jour"):
        st.markdown("""
        **Version beta 25w58b** (Nov 2025)
        - ✅ Ajout des templates Excel téléchargeables
        - ✅ Page d'accueil avec navigation améliorée
        - ✅ Menu avec icônes personnalisées
        - ✅ Masquage du menu natif Streamlit
        
        **Version beta 25w57a** (Oct 2025)
        - ✨ Ajout de ThermoTool gamma incomplete fit
        - 🐛 Corrections diverses sur KDTool
        """)
    
    # Section contact ou aide
    with st.expander("❓ Besoin d'aide ?"):
        st.markdown("""
        Pour toute question ou problème technique :
        
        - 📧 Email : M.Benonit@uliege.be (ou B.Cabrera@uliege.be)
        - 💬 Support : Contactez Maxime (ou Bastien) via teams
        - 📥 Templates : Téléchargez les fichiers exemples ci-dessus
        - 📚 Documentation : Chaque outil contient des instructions détaillées
        """)

else:
    # ============================================================================
    # CHARGEMENT DYNAMIQUE DES AUTRES PAGES
    # ============================================================================
    try:
        module_name = apps[st.session_state.page]['module']
        if module_name:
            mod = importlib.import_module(f"pages.{module_name}")
            mod.app()
        else:
            st.error("Module non défini pour cette page")
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de l'application : {e}")
        st.info("💡 Vérifiez que :")
        st.markdown("""
        - Le fichier existe dans le dossier `pages/`
        - Le fichier contient une fonction `app()`
        - Tous les imports nécessaires sont présents
        """)
        
        with st.expander("🔍 Détails de l'erreur (debug)"):
            import traceback
            st.code(traceback.format_exc())
