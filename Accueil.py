import streamlit as st
import os
import importlib

ICON_FOLDER = "assets"
# Correspondance nom logiciel / nom fichier python dans pages/ / icône SVG
apps = {
    "BDTool": {"module": "BDTool", "icon": "icone_BD.svg"},
    "KDTool": {"module": "KDTool", "icon": "icone_KD.svg"},
    "MassCalc": {"module": "MassCalc", "icon": "icone_MC.svg"},
    "ThermoTool statistic fit": {"module": "ThermoTool_statistic_fit", "icon": "icone_TT.svg"},
    "ThermoTool gamma incomplete fit": {"module": "ThermoTool_gamma_incomplete_fit", "icon": "icone_TT.svg"},
    # Ajoutez vos apps ici suivant la structure pages/vos_fichiers.py
}

# Initial page
if 'page' not in st.session_state:
    st.session_state.page = list(apps.keys())[0]

st.sidebar.title("MS Lab Toolbox")

for app_name, info in apps.items():
    icon_path = os.path.join(ICON_FOLDER, info["icon"])
    cols = st.sidebar.columns([1, 4], gap="small")
    with cols[0]:
        if os.path.exists(icon_path):
            st.image(icon_path, width=24)
        else:
            st.write("❓")
    with cols[1]:
        if st.button(app_name):
            st.session_state.page = app_name

st.title(f"Application : {st.session_state.page}")

# Import dynamique du module de l'app sélectionnée
try:
    module_name = f"pages.{apps[st.session_state.page]['module']}"
    mod = importlib.import_module(module_name)
    mod.app()  # Supposant que chaque page a une fonction app() qui lance l'app
except Exception as e:
    st.error(f"Erreur chargement app : {e}")
