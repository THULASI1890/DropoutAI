# ============================================================
# STUDENT DROPOUT PREDICTION AI
# Full Streamlit Application
# XGBoost Model
# ============================================================

import os
import pickle
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dropout AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "dropout_model.pkl"

HISTORY_FILE = "prediction_history.csv"

USERS_FILE = "users.csv"

ACCESS_EMAIL = "admin@dropoutai.com"


# ============================================================
# EXACT FEATURES USED BY YOUR XGBOOST MODEL
# ============================================================

MODEL_FEATURES = [
    "grade_level",
    "attendance_rate",
    "attendance_trend",
    "gpa",
    "gpa_trend",
    "assignment_completion_rate",
    "assignment_velocity",
    "late_assignments",
    "failed_subjects",
    "exam_average",
    "exam_trend",
    "lms_login_frequency",
    "disciplinary_incidents"
]


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):

    st.error(
        f"❌ `{MODEL_FILE}` was not found."
    )

    st.info(
        "Place your trained dropout_model.pkl file in the same folder as app.py."
    )

    st.stop()


try:

    with open(MODEL_FILE, "rb") as file:

        model = pickle.load(file)

except Exception as e:

    st.error("❌ Could not load the model.")

    st.exception(e)

    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

* {
    font-family: 'Inter', sans-serif;
}

.stApp {

    background:
    radial-gradient(
        circle at 10% 10%,
        rgba(37, 99, 235, 0.15),
        transparent 28%
    ),
    radial-gradient(
        circle at 90% 15%,
        rgba(124, 58, 237, 0.13),
        transparent 28%
    ),
    radial-gradient(
        circle at 50% 100%,
        rgba(14, 165, 233, 0.08),
        transparent 30%
    ),
    #030303;

    color: #ffffff;

}


/* Hide Streamlit menu */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* Main content */

.block-container {

    max-width: 1450px;

    padding-top: 2rem;

    padding-bottom: 5rem;

}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {

    background:
    linear-gradient(
        180deg,
        #020617 0%,
        #030303 100%
    );

    border-right:
    1px solid rgba(255,255,255,0.08);

}


.sidebar-logo {

    text-align: center;

    padding: 20px 5px;

}

.sidebar-icon {

    font-size: 45px;

}

.sidebar-title {

    font-size: 22px;

    font-weight: 800;

}

.sidebar-subtitle {

    color: #9ca3af;

    font-size: 12px;

}


/* ============================================================
   LOGIN
   ============================================================ */

.login-wrapper {

    max-width: 470px;

    margin:
    45px auto;

}

.login-card {

    background:
    linear-gradient(
        145deg,
        rgba(17,24,39,0.98),
        rgba(3,3,3,0.98)
    );

    border:
    1px solid rgba(255,255,255,0.08);

    border-radius: 25px;

    padding: 35px;

    box-shadow:
    0 25px 80px rgba(0,0,0,0.75);

    text-align: center;

}

.login-logo {

    width: 80px;

    height: 80px;

    margin: auto;

    display: flex;

    justify-content: center;

    align-items: center;

    border-radius: 22px;

    background:
    linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );

    font-size: 40px;

    box-shadow:
    0 10px 40px rgba(37,99,235,0.35);

}

.login-title {

    font-size: 30px;

    font-weight: 800;

    margin-top: 20px;

}

.login-subtitle {

    color: #9ca3af;

    font-size: 14px;

    margin-bottom: 25px;

}


/* Access email */

.access-box {

    margin-top: 22px;

    padding: 16px;

    border-radius: 14px;

    background:
    rgba(37,99,235,0.10);

    border:
    1px solid rgba(59,130,246,0.30);

}

.access-label {

    color: #60a5fa;

    font-size: 12px;

    font-weight: 700;

    letter-spacing: 1px;

}

.access-email {

    font-weight: 800;

    margin-top: 5px;

}


/* ============================================================
   INPUTS
   ============================================================ */

.stTextInput input,
.stNumberInput input {

    background:
    #111827 !important;

    color:
    white !important;

    border:
    1px solid #374151 !important;

    border-radius:
    10px !important;

    min-height:
    44px !important;

}

.stTextInput input:focus,
.stNumberInput input:focus {

    border-color:
    #3b82f6 !important;

}

.stSelectbox div[data-baseweb="select"] {

    background:
    #111827 !important;

}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {

    min-height: 45px;

    border: none;

    border-radius: 11px;

    color: white;

    font-weight: 700;

    background:
    linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );

    transition:
    all 0.2s ease;

}

.stButton > button:hover {

    transform:
    translateY(-2px);

    box-shadow:
    0 10px 25px rgba(37,99,235,0.25);

}


/* ============================================================
   PAGE HEADINGS
   ============================================================ */

.page-title {

    font-size: 36px;

    font-weight: 800;

    letter-spacing: -1px;

}

.page-subtitle {

    color: #9ca3af;

    margin-top: 5px;

    margin-bottom: 25px;

}


/* ============================================================
   CARDS
   ============================================================ */

.card {

    background:
    linear-gradient(
        145deg,
        rgba(17,24,39,0.88),
        rgba(5,5,5,0.88)
    );

    border:
    1px solid rgba(255,255,255,0.07);

    border-radius: 18px;

    padding: 22px;

    margin-top: 18px;

}


/* ============================================================
   METRICS
   ============================================================ */

.metric-card {

    background:
    linear-gradient(
        145deg,
        rgba(17,24,39,0.96),
        rgba(5,5,5,0.96)
    );

    border:
    1px solid rgba(255,255,255,0.08);

    border-radius: 17px;

    padding: 20px;

    min-height: 120px;

}

.metric-label {

    color: #9ca3af;

    font-size: 13px;

}

.metric-value {

    font-size: 29px;

    font-weight: 800;

    margin-top: 7px;

}


/* ============================================================
   RISK RESULT
   ============================================================ */

.high-risk {

    background:
    rgba(127,29,29,0.25);

    border:
    1px solid rgba(239,68,68,0.25);

    border-left:
    6px solid #ef4444;

    border-radius: 16px;

    padding: 25px;

}

.medium-risk {

    background:
    rgba(120,53,15,0.25);

    border:
    1px solid rgba(245,158,11,0.25);

    border-left:
    6px solid #f59e0b;

    border-radius: 16px;

    padding: 25px;

}

.low-risk {

    background:
    rgba(6,78,59,0.25);

    border:
    1px solid rgba(16,185,129,0.25);

    border-left:
    6px solid #10b981;

    border-radius: 16px;

    padding: 25px;

}


/* ============================================================
   MOBILE
   ============================================================ */

@media screen and (max-width: 768px) {

    .block-container {

        padding-left:
        12px !important;

        padding-right:
        12px !important;

        padding-top:
        1rem !important;

        padding-bottom:
        90px !important;

    }

    section[data-testid="stSidebar"] {

        display:
        none;

    }

    .page-title {

        font-size:
        26px;

    }

    .page-subtitle {

        font-size:
        13px;

    }

    .metric-card {

        min-height:
        90px;

        padding:
        15px;

        margin-bottom:
        8px;

    }

    .metric-value {

        font-size:
        22px;

    }

    .card {

        padding:
        16px;

    }

    .login-wrapper {

        margin:
        10px auto;

    }

    .login-card {

        padding:
        25px 18px;

    }

}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:

    st.session_state.logged_in = False


if "page" not in st.session_state:

    st.session_state.page = "Dashboard"


if "register" not in st.session_state:

    st.session_state.register = False


if "username" not in st.session_state:

    st.session_state.username = ""


# ============================================================
# USER DATABASE FUNCTIONS
# ============================================================

def hash_password(password):

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


def load_users():

    if not os.path.exists(USERS_FILE):

        df = pd.DataFrame(
            columns=[
                "Name",
                "Email",
                "Password"
            ]
        )

        df.to_csv(
            USERS_FILE,
            index=False
        )

    return pd.read_csv(
        USERS_FILE
    )


def register_user(
    name,
    email,
    password
):

    users = load_users()

    if not users.empty:

        existing = users[
            users["Email"]
            .astype(str)
            .str.lower()
            == email.lower()
        ]

        if not existing.empty:

            return False


    new_user = pd.DataFrame({

        "Name": [
            name
        ],

        "Email": [
            email
        ],

        "Password": [
            hash_password(password)
        ]

    })


    users = pd.concat(
        [
            users,
            new_user
        ],
        ignore_index=True
    )


    users.to_csv(
        USERS_FILE,
        index=False
    )


    return True


def authenticate(
    email,
    password
):

    users = load_users()

    if users.empty:

        return False


    password_hash = hash_password(
        password
    )


    result = users[

        (
            users["Email"]
            .astype(str)
            .str.lower()
            == email.lower()
        )

        &

        (
            users["Password"]
            == password_hash
        )

    ]


    return len(result) > 0


# ============================================================
# HISTORY FUNCTIONS
# ============================================================

def load_history():

    if not os.path.exists(HISTORY_FILE):

        return pd.DataFrame()


    try:

        return pd.read_csv(
            HISTORY_FILE
        )

    except Exception:

        return pd.DataFrame()


def save_prediction(record):

    history = load_history()

    new_record = pd.DataFrame(
        [record]
    )


    history = pd.concat(
        [
            history,
            new_record
        ],
        ignore_index=True
    )


    history.to_csv(
        HISTORY_FILE,
        index=False
    )


# ============================================================
# RISK FUNCTION
# ============================================================

def get_risk_level(probability):

    if probability >= 70:

        return "High Risk"

    elif probability >= 40:

        return "Medium Risk"

    else:

        return "Low Risk"


# ============================================================
# LOGIN PAGE
# ============================================================

if not st.session_state.logged_in:

    st.markdown(
        '<div class="login-wrapper">',
        unsafe_allow_html=True
    )


    # ========================================================
    # REGISTER PAGE
    # ========================================================

    if st.session_state.register:

        st.markdown(
            """
            <div class="login-card">

                <div class="login-logo">
                    🎓
                </div>

                <div class="login-title">
                    Create Account
                </div>

                <div class="login-subtitle">
                    Join Dropout Prediction AI
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        with st.form(
            "register_form"
        ):

            name = st.text_input(
                "Full Name"
            )

            email = st.text_input(
                "Email Address"
            )

            password = st.text_input(
                "Password",
                type="password"
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password"
            )


            create_account = st.form_submit_button(
                "✨ Create Account",
                use_container_width=True
            )


        if create_account:

            if not name or not email or not password:

                st.error(
                    "Please fill in all fields."
                )

            elif "@" not in email:

                st.error(
                    "Please enter a valid email address."
                )

            elif password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            elif len(password) < 6:

                st.error(
                    "Password must contain at least 6 characters."
                )

            else:

                success = register_user(
                    name,
                    email,
                    password
                )


                if success:

                    st.success(
                        "✅ Account created successfully."
                    )

                    st.session_state.register = False

                    st.rerun()

                else:

                    st.error(
                        "An account with this email already exists."
                    )


        if st.button(
            "← Back to Login",
            use_container_width=True
        ):

            st.session_state.register = False

            st.rerun()


    # ========================================================
    # LOGIN
    # ========================================================

    else:

        st.markdown(
            """
            <div class="login-card">

                <div class="login-logo">
                    🎓
                </div>

                <div class="login-title">
                    Dropout Prediction AI
                </div>

                <div class="login-subtitle">
                    AI-powered student dropout risk analytics
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        with st.form(
            "login_form"
        ):

            email = st.text_input(
                "Email Address",
                placeholder="Enter your email"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )


            login = st.form_submit_button(
                "🔐 Login",
                use_container_width=True
            )


        if login:

            if authenticate(
                email,
                password
            ):

                st.session_state.logged_in = True

                st.session_state.username = email

                st.session_state.page = "Dashboard"

                st.rerun()

            else:

                st.error(
                    "❌ Invalid email or password."
                )


        # Access email

        st.markdown(
            f"""
            <div class="access-box">

                <div class="access-label">
                    🔑 WEBSITE ACCESS EMAIL
                </div>

                <div class="access-email">
                    {ACCESS_EMAIL}
                </div>

                <div style="
                    color:#9ca3af;
                    font-size:12px;
                    margin-top:5px;
                ">
                    Use this email when requesting access.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.write("")


        if st.button(
            "✨ Create New Account",
            use_container_width=True
        ):

            st.session_state.register = True

            st.rerun()


    st.markdown(
        """
        <div style="
            text-align:center;
            color:#6b7280;
            margin-top:35px;
            font-size:13px;
        ">

            🎓 Dropout Prediction AI

            <br>

            Machine Learning • Early Intervention • Student Success

            <br><br>

            © 2026

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-logo">

            <div class="sidebar-icon">
                🎓
            </div>

            <div class="sidebar-title">
                Dropout AI
            </div>

            <div class="sidebar-subtitle">
                Student Risk Analytics
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("---")


    st.write(
        f"👤 **{st.session_state.username}**"
    )


    st.markdown("---")


    if st.button(
        "📊 Dashboard",
        use_container_width=True
    ):

        st.session_state.page = "Dashboard"

        st.rerun()


    if st.button(
        "🔮 Predict Dropout",
        use_container_width=True
    ):

        st.session_state.page = "Predict"

        st.rerun()


    if st.button(
        "🧾 Prediction History",
        use_container_width=True
    ):

        st.session_state.page = "History"

        st.rerun()


    st.markdown("---")


    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False

        st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    st.markdown(
        """
        <div class="page-title">
            📊 Student Dropout Dashboard
        </div>

        <div class="page-subtitle">
            Monitor student risk predictions and identify students
            who may need early intervention.
        </div>
        """,
        unsafe_allow_html=True
    )


    history = load_history()


    total_predictions = len(history)


    if total_predictions > 0:

        high_risk = len(
            history[
                history["Risk Level"]
                == "High Risk"
            ]
        )

        medium_risk = len(
            history[
                history["Risk Level"]
                == "Medium Risk"
            ]
        )

        low_risk = len(
            history[
                history["Risk Level"]
                == "Low Risk"
            ]
        )

        average_risk = history[
            "Probability"
        ].mean()

    else:

        high_risk = 0

        medium_risk = 0

        low_risk = 0

        average_risk = 0


    # ========================================================
    # METRICS
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    👥 Total Predictions
                </div>

                <div class="metric-value">
                    {total_predictions}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    🔴 High Risk
                </div>

                <div class="metric-value">
                    {high_risk}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    🟠 Medium Risk
                </div>

                <div class="metric-value">
                    {medium_risk}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c4:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    🎯 Average Risk
                </div>

                <div class="metric-value">
                    {average_risk:.1f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # CHARTS
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### 📈 Risk Distribution"
        )


        if total_predictions > 0:

            chart = pd.DataFrame({

                "Risk Level": [
                    "High Risk",
                    "Medium Risk",
                    "Low Risk"
                ],

                "Students": [
                    high_risk,
                    medium_risk,
                    low_risk
                ]

            })


            st.bar_chart(
                chart.set_index(
                    "Risk Level"
                )
            )

        else:

            st.info(
                "No prediction data yet."
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### ⚡ Quick Actions"
        )

        st.write(
            "Use the trained XGBoost model to predict a student's dropout risk."
        )


        if st.button(
            "🔮 Make New Prediction",
            use_container_width=True
        ):

            st.session_state.page = "Predict"

            st.rerun()


        st.write("")


        if st.button(
            "🧾 View Prediction History",
            use_container_width=True
        ):

            st.session_state.page = "History"

            st.rerun()


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ========================================================
    # RECENT PREDICTIONS
    # ========================================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        "### 🕒 Recent Predictions"
    )


    if total_predictions > 0:

        st.dataframe(
            history.tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Your recent predictions will appear here."
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif st.session_state.page == "Predict":

    st.markdown(
        """
        <div class="page-title">
            🔮 Dropout Risk Prediction
        </div>

        <div class="page-subtitle">
            Enter the student's academic information and generate
            an AI-powered dropout risk prediction.
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # STUDENT INFORMATION
    # ========================================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        "### 👨‍🎓 Student Information"
    )


    c1, c2 = st.columns(2)


    with c1:

        student_name = st.text_input(
            "Student Name",
            placeholder="Enter student name"
        )


    with c2:

        student_id = st.text_input(
            "Student ID",
            placeholder="Enter student ID"
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # ========================================================
    # EXACT 13 MODEL INPUTS
    # ========================================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        "### 📚 Academic & Behavioural Information"
    )


    c1, c2, c3 = st.columns(3)


    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with c1:

        grade_level = st.number_input(
            "Grade Level",
            min_value=1,
            max_value=20,
            value=10,
            step=1
        )


        attendance_rate = st.number_input(
            "Attendance Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.1
        )


        attendance_trend = st.number_input(
            "Attendance Trend",
            min_value=-100.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            help="Change in attendance compared with previous period."
        )


        gpa = st.number_input(
            "GPA",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )


        gpa_trend = st.number_input(
            "GPA Trend",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="Change in GPA compared with previous period."
        )


    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with c2:

        assignment_completion_rate = st.number_input(
            "Assignment Completion Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.1
        )


        assignment_velocity = st.number_input(
            "Assignment Velocity",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1
        )


        late_assignments = st.number_input(
            "Late Assignments",
            min_value=0,
            value=0,
            step=1
        )


        failed_subjects = st.number_input(
            "Failed Subjects",
            min_value=0,
            value=0,
            step=1
        )


    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with c3:

        exam_average = st.number_input(
            "Exam Average (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.1
        )


        exam_trend = st.number_input(
            "Exam Trend",
            min_value=-100.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            help="Change in examination performance."
        )


        lms_login_frequency = st.number_input(
            "LMS Login Frequency",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1
        )


        disciplinary_incidents = st.number_input(
            "Disciplinary Incidents",
            min_value=0,
            value=0,
            step=1
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button(
        "🚀 PREDICT DROPOUT RISK",
        use_container_width=True
    ):

        if not student_name.strip():

            st.warning(
                "Please enter the student name."
            )

            st.stop()


        # ====================================================
        # CREATE INPUT DATA
        # ====================================================

        input_data = pd.DataFrame({

            "grade_level": [
                grade_level
            ],

            "attendance_rate": [
                attendance_rate
            ],

            "attendance_trend": [
                attendance_trend
            ],

            "gpa": [
                gpa
            ],

            "gpa_trend": [
                gpa_trend
            ],

            "assignment_completion_rate": [
                assignment_completion_rate
            ],

            "assignment_velocity": [
                assignment_velocity
            ],

            "late_assignments": [
                late_assignments
            ],

            "failed_subjects": [
                failed_subjects
            ],

            "exam_average": [
                exam_average
            ],

            "exam_trend": [
                exam_trend
            ],

            "lms_login_frequency": [
                lms_login_frequency
            ],

            "disciplinary_incidents": [
                disciplinary_incidents
            ]

        })


        # ====================================================
        # FORCE EXACT FEATURE ORDER
        # ====================================================

        input_data = input_data[
            MODEL_FEATURES
        ]


        try:

            # =================================================
            # MODEL PREDICTION
            # =================================================

            prediction = model.predict(
                input_data
            )[0]


            # =================================================
            # PREDICTION PROBABILITY
            # =================================================

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    input_data
                )[0]


                if hasattr(
                    model,
                    "classes_"
                ):

                    classes = list(
                        model.classes_
                    )


                    if 1 in classes:

                        dropout_index = classes.index(
                            1
                        )

                    elif "Dropout" in classes:

                        dropout_index = classes.index(
                            "Dropout"
                        )

                    else:

                        dropout_index = (
                            len(probabilities) - 1
                        )

                else:

                    dropout_index = (
                        len(probabilities) - 1
                    )


                probability = (
                    float(
                        probabilities[
                            dropout_index
                        ]
                    ) * 100
                )


            else:

                probability = (
                    100.0
                    if prediction == 1
                    else 0.0
                )


            probability = max(
                0,
                min(
                    100,
                    probability
                )
            )


            # =================================================
            # RISK LEVEL
            # =================================================

            risk = get_risk_level(
                probability
            )


            # =================================================
            # PREDICTION RESULT
            # =================================================

            st.markdown(
                "## 🎯 Prediction Result"
            )


            if risk == "High Risk":

                st.markdown(
                    f"""
                    <div class="high-risk">

                        <h2>🔴 HIGH DROPOUT RISK</h2>

                        <h3>
                            {student_name}
                        </h3>

                        <p>
                            Student ID:
                            {student_id if student_id else "Not provided"}
                        </p>

                        <h1>
                            {probability:.2f}%
                        </h1>

                        <p>
                            Estimated probability of dropout
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            elif risk == "Medium Risk":

                st.markdown(
                    f"""
                    <div class="medium-risk">

                        <h2>🟠 MEDIUM DROPOUT RISK</h2>

                        <h3>
                            {student_name}
                        </h3>

                        <p>
                            Student ID:
                            {student_id if student_id else "Not provided"}
                        </p>

                        <h1>
                            {probability:.2f}%
                        </h1>

                        <p>
                            Estimated probability of dropout
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            else:

                st.markdown(
                    f"""
                    <div class="low-risk">

                        <h2>🟢 LOW DROPOUT RISK</h2>

                        <h3>
                            {student_name}
                        </h3>

                        <p>
                            Student ID:
                            {student_id if student_id else "Not provided"}
                        </p>

                        <h1>
                            {probability:.2f}%
                        </h1>

                        <p>
                            Estimated probability of dropout
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # =================================================
            # RISK FACTORS
            # =================================================

            risk_factors = []


            if attendance_rate < 75:

                risk_factors.append(
                    "Low attendance rate"
                )


            if attendance_trend < -5:

                risk_factors.append(
                    "Attendance is declining"
                )


            if gpa < 5:

                risk_factors.append(
                    "Low GPA"
                )


            if gpa_trend < -0.5:

                risk_factors.append(
                    "GPA is declining"
                )


            if assignment_completion_rate < 70:

                risk_factors.append(
                    "Low assignment completion"
                )


            if assignment_velocity < 3:

                risk_factors.append(
                    "Low assignment activity"
                )


            if late_assignments >= 3:

                risk_factors.append(
                    "Multiple late assignments"
                )


            if failed_subjects >= 2:

                risk_factors.append(
                    "Multiple failed subjects"
                )


            if exam_average < 50:

                risk_factors.append(
                    "Low exam performance"
                )


            if exam_trend < -5:

                risk_factors.append(
                    "Exam performance is declining"
                )


            if lms_login_frequency < 3:

                risk_factors.append(
                    "Low LMS engagement"
                )


            if disciplinary_incidents >= 2:

                risk_factors.append(
                    "Multiple disciplinary incidents"
                )


            # =================================================
            # RECOMMENDATIONS
            # =================================================

            recommendations = []


            if attendance_rate < 75:

                recommendations.append(
                    "📅 Improve attendance and monitor weekly attendance."
                )


            if attendance_trend < -5:

                recommendations.append(
                    "📈 Contact the student because attendance is declining."
                )


            if gpa < 5:

                recommendations.append(
                    "📚 Provide academic mentoring."
                )


            if gpa_trend < -0.5:

                recommendations.append(
                    "📊 Monitor GPA and provide additional academic support."
                )


            if assignment_completion_rate < 70:

                recommendations.append(
                    "📝 Create an assignment completion plan."
                )


            if late_assignments >= 3:

                recommendations.append(
                    "⏰ Support the student with time management."
                )


            if failed_subjects >= 2:

                recommendations.append(
                    "🎓 Provide subject-specific tutoring."
                )


            if exam_average < 50:

                recommendations.append(
                    "🧠 Provide examination preparation support."
                )


            if lms_login_frequency < 3:

                recommendations.append(
                    "💻 Encourage regular LMS engagement."
                )


            if disciplinary_incidents >= 2:

                recommendations.append(
                    "🤝 Arrange student counselling/support."
                )


            if not recommendations:

                recommendations.append(
                    "✅ Student currently shows healthy academic indicators."
                )


            # =================================================
            # DISPLAY FACTORS
            # =================================================

            c1, c2 = st.columns(2)


            with c1:

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    "### ⚠️ Risk Factors"
                )


                if risk_factors:

                    for factor in risk_factors:

                        st.write(
                            "• " + factor
                        )

                else:

                    st.success(
                        "No major risk indicators detected."
                    )


                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


            with c2:

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    "### 💡 Recommended Actions"
                )


                for recommendation in recommendations:

                    st.write(
                        recommendation
                    )


                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


            # =================================================
            # SAVE PREDICTION
            # =================================================

            record = {

                "Date":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "User":
                    st.session_state.username,

                "Student Name":
                    student_name,

                "Student ID":
                    student_id,

                "Prediction":
                    "Dropout"
                    if prediction == 1
                    else "No Dropout",

                "Risk Level":
                    risk,

                "Probability":
                    round(
                        probability,
                        2
                    ),

                "Grade Level":
                    grade_level,

                "Attendance Rate":
                    attendance_rate,

                "Attendance Trend":
                    attendance_trend,

                "GPA":
                    gpa,

                "GPA Trend":
                    gpa_trend,

                "Assignment Completion Rate":
                    assignment_completion_rate,

                "Assignment Velocity":
                    assignment_velocity,

                "Late Assignments":
                    late_assignments,

                "Failed Subjects":
                    failed_subjects,

                "Exam Average":
                    exam_average,

                "Exam Trend":
                    exam_trend,

                "LMS Login Frequency":
                    lms_login_frequency,

                "Disciplinary Incidents":
                    disciplinary_incidents

            }


            save_prediction(
                record
            )


            st.success(
                "✅ Prediction successfully saved to history."
            )


        except Exception as e:

            st.error(
                "❌ Prediction could not be completed."
            )

            st.error(
                "Please check that the model is the same model trained with the 13 features used by this application."
            )

            st.exception(e)


# ============================================================
# HISTORY PAGE
# ============================================================

elif st.session_state.page == "History":

    st.markdown(
        """
        <div class="page-title">
            🧾 Prediction History
        </div>

        <div class="page-subtitle">
            Review previous student dropout predictions.
        </div>
        """,
        unsafe_allow_html=True
    )


    history = load_history()


    if history.empty:

        st.info(
            "No prediction history available yet."
        )

    else:

        # ====================================================
        # FILTERS
        # ====================================================

        c1, c2 = st.columns(2)


        with c1:

            risk_filter = st.selectbox(
                "Risk Level",
                [
                    "All",
                    "High Risk",
                    "Medium Risk",
                    "Low Risk"
                ]
            )


        with c2:

            search = st.text_input(
                "🔎 Search Student"
            )


        filtered = history.copy()


        if risk_filter != "All":

            filtered = filtered[
                filtered["Risk Level"]
                == risk_filter
            ]


        if search:

            filtered = filtered[
                filtered[
                    "Student Name"
                ]
                .astype(str)
                .str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]


        # ====================================================
        # SUMMARY
        # ====================================================

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )


        st.markdown(
            "### 📊 Filtered Results"
        )


        st.write(
            f"Showing **{len(filtered)}** prediction(s)."
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # ====================================================
        # TABLE
        # ====================================================

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )


        st.markdown(
            "### 📋 Prediction Records"
        )


        if not filtered.empty:

            st.dataframe(
                filtered.iloc[::-1],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No records match your filter."
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # ====================================================
        # DOWNLOAD
        # ====================================================

        csv_data = filtered.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            label="📥 Download Prediction History",
            data=csv_data,
            file_name="dropout_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# MOBILE NAVIGATION
# ============================================================

if st.session_state.logged_in:

    st.markdown(
        """
        <style>

        @media screen and (max-width:768px) {

            .mobile-nav-spacer {

                height: 70px;

            }

        }

        </style>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="mobile-nav-spacer"></div>',
        unsafe_allow_html=True
    )


    b1, b2, b3, b4 = st.columns(4)


    with b1:

        if st.button(
            "🏠 Home",
            key="mobile_home",
            use_container_width=True
        ):

            st.session_state.page = "Dashboard"

            st.rerun()


    with b2:

        if st.button(
            "🔮 Predict",
            key="mobile_predict",
            use_container_width=True
        ):

            st.session_state.page = "Predict"

            st.rerun()


    with b3:

        if st.button(
            "🧾 History",
            key="mobile_history",
            use_container_width=True
        ):

            st.session_state.page = "History"

            st.rerun()


    with b4:

        if st.button(
            "🚪 Logout",
            key="mobile_logout",
            use_container_width=True
        ):

            st.session_state.logged_in = False

            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#6b7280;
        padding:30px 10px;
        margin-top:30px;
        font-size:13px;
    ">

        🎓 <b>Dropout Prediction AI</b>

        <br>

        XGBoost Student Risk Analytics

        <br>

        Early Detection • Prediction • Intervention

        <br><br>

        © 2026

    </div>
    """,
    unsafe_allow_html=True
)