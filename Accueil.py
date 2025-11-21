import streamlit as st
import os
import importlib

# Cacher le menu de navigation natif Streamlit
st.markdown("""
    <style>
        /* Cacher le menu de navigation par défaut */
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

ICON_FOLDER = "assets"

# Correspondance nom logiciel / nom fichier python dans pages/ / icône SVG
apps = {
    "🏠 Accueil": {"module": None, "icon": "TB_logo.svg"},
    "BDTool": {"module": "BDTool", "icon": "icone_BD.svg"},
    "KDTool": {"module": "KDTool", "icon": "icone_KD.svg"},
    "MassCalc": {"module": "MassCalc", "icon": "icone_MC.svg"},
    "ThermoTool statistic fit": {"module": "ThermoTool_statistic_fit", "icon": "icone_TT.svg"},
    "ThermoTool gamma incomplete fit": {"module": "ThermoTool_gamma_incomplete_fit", "icon": "icone_TT.svg"},
}

# Initialisation sur la page d'accueil par défaut
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Accueil"

st.sidebar.title("MS Lab Toolbox")

# Menu avec icônes - VERSION AVEC BOUTONS HTML PERSONNALISÉS
for app_name, info in apps.items():
    icon_path = os.path.join(ICON_FOLDER, info["icon"])
    
    # Créer un conteneur cliquable en HTML
    is_active = st.session_state.page == app_name
    
    # Lire l'icône et l'encoder en base64 pour l'inclure dans le HTML
    icon_html = ""
    if os.path.exists(icon_path):
        with open(icon_path, 'rb') as f:
            import base64
            icon_data = base64.b64encode(f.read()).decode()
            icon_html = f'<img src="data:image/svg+xml;base64,{icon_data}" width="36" height="36" style="display: block;">'
    else:
        icon_html = '<div style="font-size: 24px;">❓</div>'
    
    # Créer un bouton HTML personnalisé avec alignement parfait
    button_style = """
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 12px;
        border-radius: 8px;
        background: """ + ('#e0f2fe' if is_active else 'transparent') + """;
        border: 1px solid """ + ('#0284c7' if is_active else 'transparent') + """;
        cursor: pointer;
        transition: all 0.2s;
        margin-bottom: 4px;
        text-decoration: none;
        color: inherit;
    """
    
    button_html = f"""
        <div style="{button_style}" onmouseover="this.style.background='#f1f5f9'" onmouseout="this.style.background='{'#e0f2fe' if is_active else 'transparent'}'">
            <div style="flex-shrink: 0; display: flex; align-items: center; justify-content: center;">
                {icon_html}
            </div>
            <div style="flex: 1; font-size: 14px; font-weight: {'600' if is_active else '400'};">
                {app_name}
            </div>
        </div>
    """
    
    # Afficher le bouton et gérer le clic
    if st.sidebar.button(f"select_{app_name}", key=f"btn_{app_name}", label_visibility="collapsed"):
        st.session_state.page = app_name
        st.rerun()
    
    # Afficher le bouton HTML au-dessus du bouton invisible Streamlit
    st.sidebar.markdown(button_html, unsafe_allow_html=True)
    st.sidebar.markdown("<div style='margin-top: -50px;'></div>", unsafe_allow_html=True)

# Affichage du contenu selon la page sélectionnée
if st.session_state.page == "🏠 Accueil":
    # PAGE D'ACCUEIL
    st.image("images/MSTB.png", width=200)
    st.title("🔬 MS Lab Toolbox")
    st.markdown("---")
    
    st.markdown("""
    ## Bienvenue dans la boîte à outils du laboratoire de spectrométrie de masse !
    
    Cette application regroupe plusieurs outils utiles pour l'analyse et le traitement de données MS.
    
    ### 📚 Outils disponibles :
    
    - **BDTool** : Visualisation et analyse de courbes de dissociation (breakdown curves)
    - **KDTool** : Détermination de constantes de dissociation (méthodes Hill & Gabelica)
    - **MassCalc** : Calculateur de masse moléculaire et m/z
    - **ThermoTool** : Analyse thermodynamique avec ajustements statistiques et gamma
    
    ### 🚀 Pour commencer :
    
    1. Sélectionnez un outil dans le menu de gauche
    2. Suivez les instructions spécifiques à chaque outil
    3. Téléchargez vos résultats en fin d'analyse
    
    ### 📖 Instructions générales :
    
    - Les fichiers CSV doivent respecter le format indiqué dans chaque outil
    - Les résultats peuvent être exportés en Excel, SVG ou PDF selon l'outil
    - Encodage recommandé : UTF-8
    - Pour toute question, contactez l'équipe du laboratoire
    
    ### ⚙️ Configuration :
    
    - **Formats supportés** : CSV, Excel, TXT
    - **Langues disponibles** : Français / English (selon l'outil)
    - **Instruments** : Compatibilité avec la plupart des spectromètres de masse
    
    ---
    
    💡 **Astuce** : Certains outils proposent des exemples de données pour vous familiariser avec les fonctionnalités.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🧪 **Instruments**\n\nTWIM, Orbitrap, Q-TOF, MALDI")
    
    with col2:
        st.success("📊 **Formats**\n\nCSV, Excel, TXT, SVG")
    
    with col3:
        st.warning("🔄 **Version**\n\n2.0 - Novembre 2025")
    
    with st.expander("📋 Dernières mises à jour"):
        st.markdown("""
        **Version 2.0** (Nov 2025)
        - ✅ Navigation optimale avec icônes alignées
        - ✅ Interface améliorée
        """)

else:
    # CHARGEMENT DYNAMIQUE DES AUTRES PAGES
    try:
        module_name = apps[st.session_state.page]['module']
        if module_name:
            mod = importlib.import_module(f"pages.{module_name}")
            mod.app()
        else:
            st.error("Module non défini pour cette page")
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
