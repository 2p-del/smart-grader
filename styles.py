"""Load the editable SmartGrader visual theme from theme.css."""

from pathlib import Path

import streamlit as st


def inject_styles() -> None:
    """Apply the app's visual-only CSS, if the theme file is available."""
    css_path = Path(__file__).with_name("theme.css")
    if css_path.is_file():
        st.markdown(f"<style>\n{css_path.read_text(encoding='utf-8')}\n</style>", unsafe_allow_html=True)
