import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Student Risk Predictor", page_icon="🎓")

# ---------- Load the saved models ----------
BASE = Path(__file__).parent

@st.cache_resource
def load_models():
    model = joblib.load(BASE / "app_model.pkl")
    scaler = joblib.load(BASE / "cluster_scaler.pkl")
    km = joblib.load(BASE / "kmeans.pkl")
    return model, scaler, km

app_model, scaler, km = load_models()

APP_COLS = ["studytime", "failures", "absences", "Dalc", "Walc", "goout", "G1", "G2"]
CLUSTER_COLS = APP_COLS + ["G3"]

# ---------- Name the clusters automatically ----------
# Look at each cluster's average G3: highest = on track, lowest = struggling
centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=CLUSTER_COLS)
order = centers["G3"].sort_values().index.tolist()
names = {
    order[0]: "Academically struggling",
    order[1]: "Lifestyle risk",
    order[-1]: "On track",
}
advice = {
    "Academically struggling": "Grades are low and falling. Consider tutoring and regular grade check-ins.",
    "Lifestyle risk": "Grades are middling but habits (absences, alcohol, going out) are a concern. Consider counselling and attendance follow-up.",
    "On track": "Doing well. Keep monitoring, no special action needed.",
}

# ---------- Page layout ----------
st.title("🎓 Student Performance & Risk Predictor")
st.write("Enter a student's details to predict their final mark (G3) and risk group.")

col1, col2 = st.columns(2)
with col1:
    studytime = st.slider("Weekly study time (1 = low, 4 = high)", 1, 4, 2)
    failures = st.slider("Past class failures", 0, 3, 0)
    absences = st.slider("Absences", 0, 75, 4)
    Dalc = st.slider("Weekday alcohol use (1 = very low, 5 = very high)", 1, 5, 1)
with col2:
    Walc = st.slider("Weekend alcohol use (1 = very low, 5 = very high)", 1, 5, 2)
    goout = st.slider("Going out with friends (1 = low, 5 = high)", 1, 5, 3)
    G1 = st.slider("Period 1 grade (G1, 0-20)", 0, 20, 10)
    G2 = st.slider("Period 2 grade (G2, 0-20)", 0, 20, 10)

# ---------- Predict ----------
if st.button("Predict"):
    student = pd.DataFrame(
        [[studytime, failures, absences, Dalc, Walc, goout, G1, G2]], columns=APP_COLS
    )

    # Step 1: predict the final mark, keep it inside 0-20
    g3_pred = float(app_model.predict(student)[0])
    g3_pred = max(0.0, min(20.0, g3_pred))

    # Step 2: add the predicted G3, scale, and find the cluster
    student["G3"] = g3_pred
    scaled = scaler.transform(student[CLUSTER_COLS])
    group = names[int(km.predict(scaled)[0])]

    # Step 3: show results
    st.subheader("Results")
    st.metric("Predicted final mark (G3)", f"{g3_pred:.1f} / 20")
    st.write(f"**Risk group:** {group}")
    st.info(advice[group])

    # Habit flags: simple rules based on the Lifestyle-risk cluster profile
    flags = []
    if Dalc >= 3:
        flags.append("high weekday alcohol use")
    if Walc >= 4:
        flags.append("high weekend alcohol use")
    if goout >= 4:
        flags.append("goes out a lot")
    if absences >= 8:
        flags.append("many absences")
    if studytime == 1:
        flags.append("low study time")

    if flags:
        st.warning(
            "Habit flags: " + ", ".join(flags) + ". "
            "These barely change the predicted mark, but they are the pattern "
            "seen in the Lifestyle-risk group."
        )
    else:
        st.success("No habit flags raised.")

st.caption(
    "Trained on 395 maths students from Portuguese schools (UCI dataset). "
    "Predictions are estimates and may not apply to other regions."
)