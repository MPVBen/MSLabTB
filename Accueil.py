import streamlit as st
import os

# Dictionnaire : nom page -> icône (SVG dans assets) + fonction d'affichage
pages = {
    "Accueil": {"icon": "assets/TB_logo.svg", "page": "pages/Accueil.py"},
    "BDTool": {"icon": "assets/icone_BD.svg", "page": "pages/BDTool.py"},
    "KDTool": {"icon": "assets/icone_KD.svg", "page": "pages/KDTool.py"},
    "MassCalc": {"icon": "assets/icone_MC.svg", "page": "pages/MassCalc.py"},
    "ThermoTool logistic fit": {"icon": "assets/icone_TT.svg", "page": "pages/ThermoTool_statistic_fit.py"},
    "ThermoTool gamma incomplete fit": {"icon": "assets/icone_TT.svg", "page": "pages/ThermoTool_gamma_incomplete_fit.py"},
}

if 'page' not in st.session_state:
    st.session_state.page = "Accueil"

st.sidebar.title("Menu")

def page_button(name, icon_path):
    cols = st.sidebar.columns([1, 4], gap="small")
    with cols[0]:
        st.image(icon_path, width=24)
    with cols[1]:
        if st.button(name):
            st.session_state.page = name

for page_name, page_info in pages.items():
    if os.path.exists(page_info["icon"]):
        page_button(page_name, page_info["icon"])
    else:
        st.sidebar.write(f"(Icone manquante: {page_name})")

# Charge la page choisie via st.experimental_get_pages (Streamlit multipage)
# Streamlit multi-page app gère le changement automatiquement si structure pages/
st.write(f"Page sélectionnée : **{st.session_state.page}**")

# Optionnel: pour exécuter dynamiquement une page (si app 1 fichier)
# else laissez Streamlit multipage gérer automatiquement
