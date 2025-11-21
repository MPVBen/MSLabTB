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

# Menu avec icônes - ALIGNEMENT PARFAIT
for app_name, info in apps.items():
    icon_path = os.path.join(ICON_FOLDER, info["icon"])
    
    # Créer un conteneur avec colonnes
    cols = st.sidebar.columns([1, 5], gap="small")
    
    with cols[0]:
        # Afficher l'icône avec un conteneur HTML pour centrage vertical
        if os.path.exists(icon_path):
            # Wrapper HTML pour aligner verticalement au centre du bouton
            st.markdown("""
                <div style="display: flex; align-items: center; justify-content: center; height: 42px;">
            """, unsafe_allow_html=True)
            st.image(icon_path, width=44)  # Légèrement plus grand que le bouton
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="display: flex; align-items: center; justify-content: center; height: 42px; font-size: 24px;">
                    ❓
                </div>
            """, unsafe_allow_html=True)
    
    with cols[1]:
        # Bouton avec hauteur cohérente
        if st.button(app_name, key=f"btn_{app_name}", use_container_width=True):
            st.session_state.page = app_name

# Affichage du contenu selon la page sélectionnée
if st.session_state.page == "🏠 Accueil":
    # ============================================================================
    # PAGE D'ACCUEIL
    # ============================================================================
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
    
    # Section avec colonnes d'information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🧪 **Instruments**\n\nTWIM, Orbitrap, Q-TOF, MALDI")
    
    with col2:
        st.success("📊 **Formats**\n\nCSV, Excel, TXT, SVG")
    
    with col3:
        st.warning("🔄 **Version**\n\n2.0 - Novembre 2025")
    
    # Section optionnelle : Changelog ou actualités
    with st.expander("📋 Dernières mises à jour"):
        st.markdown("""
        **Version 2.0** (Nov 2025)
        - ✅ Ajout de la page d'accueil avec navigation améliorée
        - ✅ Menu avec icônes personnalisées parfaitement alignées
        - ✅ Masquage du menu natif Streamlit
        - ✅ Amélioration de l'interface utilisateur
        
        **Version 1.5** (Oct 2025)
        - ✨ Ajout de ThermoTool gamma incomplete fit
        - 🐛 Corrections diverses sur KDTool
        """)
    
    # Section contact ou aide
    with st.expander("❓ Besoin d'aide ?"):
        st.markdown("""
        Pour toute question ou problème technique :
        
        - 📧 Email : mslab@universite.be
        - 💬 Support : Contactez l'équipe du laboratoire
        - 📚 Documentation : Consultez les manuels de chaque outil
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
        
        # Afficher plus de détails en mode debug
        with st.expander("🔍 Détails de l'erreur (debug)"):
            import traceback
            st.code(traceback.format_exc())
