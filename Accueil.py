import streamlit as st

# Chemins vers vos icônes SVG (mettez vos propres fichiers dans un dossier 'assets')
icons = {
    "Accueil": "assets/TB_logo.svg",
    "BDTool": "assets/icone_BD.svg",
    "KDTool": "assets/icone_KD.svg",
    "MassCalc": "assets/icone_MC.svg",
    "ThermoTool logistic fit": "assets/icone_TT.svg",
    "ThermoTool gamma incomplete fit": "assets/icone_TT.svg",
}

# Initialisation de la sélection
if 'page' not in st.session_state:
    st.session_state.page = "Accueil"

st.sidebar.title("Menu")

# Affichage du menu avec icônes SVG
for page, icon_path in icons.items():
    # Affiche l’icône + label avec un bouton
    col1, col2 = st.sidebar.columns([1, 4], gap="small")
    with col1:
        st.image(icon_path, width=24)
    with col2:
        if st.sidebar.button(page):
            st.session_state.page = page

# Affichage de la page sélectionnée
st.write(f"Vous êtes sur la page : **{st.session_state.page}**")
