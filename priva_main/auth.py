import streamlit as st
import re
from priva_main.config import PASSWORD_SALT

def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "login"

def is_valid_email(email):
    return re.match(r"^[\w\.\+\-]+@[a-zA-Z0-9\.\-]+\.[a-zA-Z]{2,}$", email) is not None

def is_valid_password(password):
    return (
        len(password) >= 8 and any(c.isupper() for c in password)
        and any(c.islower() for c in password) and any(c.isdigit() for c in password)
    )

def login_page():
    from priva_main.db import User
    st.subheader("🔐 Login")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
    if submit:
        if not username or not password:
            st.error("Please enter both username and password")
        else:
            user = User.authenticate(username, password, PASSWORD_SALT)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.user_id = user['id']
                st.session_state.current_page = "dashboard"
                st.rerun()
            else:
                st.error("Invalid username or password")
    st.divider()
    if st.button("Go to Register"):
        st.session_state.current_page = "register"
        st.rerun()

def register_page():
    from priva_main.db import User
    st.subheader("📝 Register")
    with st.form("register_form", clear_on_submit=False):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Create account")
    if submit:
        if not username or not email or not password:
            st.error("All fields are required")
        elif not is_valid_email(email):
            st.error("Invalid email")
        elif not is_valid_password(password):
            st.error("Password must be ≥8 chars, include uppercase, lowercase, and a digit.")
        elif password != confirm:
            st.error("Passwords do not match")
        else:
            if User.create(username, email, password, PASSWORD_SALT):
                st.success("Account created. You can log in now.")
                st.session_state.current_page = "login"
                st.rerun()
            else:
                st.error("Username or email already exists")
    st.divider()
    if st.button("Back to Login"):
        st.session_state.current_page = "login"
        st.rerun()

def logout():
    for k in ("logged_in","username","user_id"):
        st.session_state[k] = None if k != "logged_in" else False
    st.session_state.current_page = "login"
    st.rerun()
