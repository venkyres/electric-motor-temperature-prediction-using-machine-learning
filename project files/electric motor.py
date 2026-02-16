
# ── 1. Install ───────────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "gradio", "scikit-learn", "xgboost",
                "pandas", "numpy", "matplotlib", "seaborn", "-q"])

# ── 2. Imports ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import gradio as gr

print("✅ All libraries loaded.")

# ── 3. Synthetic Dataset ─────────────────────────────────────
np.random.seed(42)
N = 5000

motor_load      = np.random.uniform(10, 100,   N)
voltage         = np.random.uniform(380, 440,  N)
current         = np.random.uniform(5,  50,    N)
ambient_temp    = np.random.uniform(15, 45,    N)
rpm             = np.random.uniform(1000, 3600, N)
vibration       = np.random.uniform(0.5, 10,   N)
humidity        = np.random.uniform(20, 90,    N)
operating_hours = np.random.uniform(0, 10000,  N)

motor_temp = (
    ambient_temp
    + 0.35 * motor_load
    + 0.15 * current
    + 0.008 * (voltage - 380)
    + 0.005 * rpm / 100
    + 1.5  * vibration
    + 0.05 * humidity
    + 0.001 * operating_hours
    + np.random.normal(0, 2, N)
)

FEATURES = ["Motor_Load_%","Voltage_V","Current_A","Ambient_Temp_C",
            "RPM","Vibration_mm_s","Humidity_%","Operating_Hours"]

df = pd.DataFrame({
    "Motor_Load_%"       : motor_load,
    "Voltage_V"          : voltage,
    "Current_A"          : current,
    "Ambient_Temp_C"     : ambient_temp,
    "RPM"                : rpm,
    "Vibration_mm_s"     : vibration,
    "Humidity_%"         : humidity,
    "Operating_Hours"    : operating_hours,
    "Motor_Temperature_C": motor_temp,
})
print(f"✅ Dataset ready  |  Temp range: {motor_temp.min():.1f}°C – {motor_temp.max():.1f}°C")

# ── 4. Train / Evaluate ──────────────────────────────────────
X = df[FEATURES]
y = df["Motor_Temperature_C"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ALL models receive scaled data → consistent and correct at inference time
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)

candidates = {
    "Linear Regression" : LinearRegression(),
    "Random Forest"     : RandomForestRegressor(n_estimators=150, max_depth=12,
                                                 random_state=42, n_jobs=-1),
    "Gradient Boosting" : GradientBoostingRegressor(n_estimators=150, learning_rate=0.08,
                                                     max_depth=5, random_state=42),
    "XGBoost"           : XGBRegressor(n_estimators=150, learning_rate=0.08,
                                        max_depth=6, random_state=42, verbosity=0),
}

results = {}
print("\n🚀 Training models…")
for name, mdl in candidates.items():
    mdl.fit(X_tr, y_train)
    pred = mdl.predict(X_te)
    results[name] = {
        "MAE" : mean_absolute_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2"  : r2_score(y_test, pred),
        "model": mdl,
    }
    print(f"  {name:<28} MAE={results[name]['MAE']:.3f}  "
          f"RMSE={results[name]['RMSE']:.3f}  R²={results[name]['R2']:.4f}")

best_name  = max(results, key=lambda k: results[k]["R2"])
best_model = results[best_name]["model"]
print(f"\n🏆 Best: {best_name}  R²={results[best_name]['R2']:.4f}")

joblib.dump(best_model, "best_motor_model.pkl")
joblib.dump(scaler,     "scaler.pkl")
print("💾 Saved model & scaler.")

# ── 5. Model Comparison Plot ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Model Comparison", fontsize=13, fontweight="bold")
names  = list(results.keys())
colors = ["#3498db","#e74c3c","#2ecc71","#f39c12"]
for ax, vals, title in zip(
        axes,
        [[results[k]["MAE"]  for k in names],
         [results[k]["RMSE"] for k in names],
         [results[k]["R2"]   for k in names]],
        ["MAE (lower=better)","RMSE (lower=better)","R² (higher=better)"]):
    ax.bar(names, vals, color=colors)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

# Feature importance from Random Forest
rf_model = results["Random Forest"]["model"]
fi = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
fi.plot(kind="barh", color="#3498db", ax=ax)
ax.set_title("Feature Importance – Random Forest", fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120, bbox_inches="tight")
plt.close()
print("✅ Plots saved.")

# ── 6. Prediction Helpers ────────────────────────────────────

def get_risk(temp: float) -> str:
    if temp < 60:  return "Normal ✅"
    if temp < 80:  return "Warning ⚠️"
    if temp < 100: return "Critical 🚨"
    return "Danger ☠️"


def scenario1_maintenance(temp: float, vibration: float, op_hours: float) -> str:
    lines = []
    if temp >= 100:
        lines += [
            "🔴 **IMMEDIATE SHUTDOWN REQUIRED**",
            "🔧 Inspect motor windings & insulation NOW",
            "🧯 Check cooling fan & ventilation path",
            "📞 Alert maintenance team immediately",
        ]
    elif temp >= 80:
        lines += [
            "🟠 **Urgent inspection within 24 hours**",
            "🔧 Clean cooling fins and check airflow",
            "🔩 Inspect bearings for abnormal wear",
            "📊 Log temperature trend hourly",
        ]
    elif temp >= 60:
        lines += [
            "🟡 **Schedule routine inspection within 1 week**",
            "🧹 Clean motor housing and vents",
            "📋 Record readings for baseline comparison",
        ]
    else:
        lines += [
            "🟢 **Motor operating within normal limits**",
            "📅 Continue with scheduled maintenance plan",
            "✅ No immediate action required",
        ]
    if vibration > 7:
        lines.append("⚡ High vibration detected – check bearing alignment")
    if op_hours > 8000:
        lines.append("🕐 High cumulative hours – plan bearing replacement soon")
    return "\n".join(f"- {l}" for l in lines)


def scenario2_energy(load: float, temp: float, voltage: float, current: float) -> str:
    power_kw  = round((voltage * current * 0.85) / 1000, 2)
    waste_pct = round(min(max(temp - 60, 0) * 0.5, 25), 1)
    if 75 <= load <= 95 and temp < 80:
        status, tip = "Optimal ✅",     "Running at peak efficiency – maintain current load."
    elif load < 50:
        status, tip = "Under-loaded ⬇️","Under-utilised; increase load for better efficiency."
    elif load > 95:
        status, tip = "Over-loaded ⬆️", "Overloaded – reduce load to prevent overheating."
    else:
        status, tip = "Sub-optimal ⚠️","Temperature elevated; inspect cooling system."
    return (
        f"- **Efficiency Status**: {status}\n"
        f"- **Current Load**: {load:.1f}%  *(Optimal: 75–95%)*\n"
        f"- **Estimated Power Draw**: {power_kw:.2f} kW\n"
        f"- **Advice**: {tip}\n"
        f"- **Potential Energy Savings**: {waste_pct}% by optimising motor load"
    )


def scenario3_reliability(temp: float, load: float,
                           vibration: float, op_hours: float) -> str:
    rul = max(0,
              15000
              - op_hours
              - max(0, (temp      - 60) / 10) * 500
              - max(0, (load      - 90) / 10) * 300
              - max(0, (vibration -  5) /  5) * 400)
    days = rul / 24
    if rul > 3000:  rel = "High 🟢"
    elif rul > 1000: rel = "Moderate 🟡"
    else:            rel = "Low 🔴 – plan replacement"
    tip = ("Plan proactive replacement before failure."
           if rul < 2000 else "Motor in good health – continue monitoring.")
    return (
        f"- **Overall Reliability**: {rel}\n"
        f"- **Estimated RUL**: {rul:,.0f} hrs  ({days:,.0f} days)\n"
        f"- **Hours Logged So Far**: {op_hours:,.0f} hrs\n"
        f"- **Vibration**: {vibration:.2f} mm/s  "
          + ("⚠️ High" if vibration > 7 else "✅ OK") + "\n"
        f"- **Recommendation**: {tip}"
    )


# ── 7. Core predict function ─────────────────────────────────

def predict(motor_load, voltage, current, ambient_temp,
            rpm, vibration, humidity, operating_hours):
    """Scales input, runs model, returns (temp_str, risk_str, markdown)."""
    row = pd.DataFrame([{
        "Motor_Load_%"   : float(motor_load),
        "Voltage_V"      : float(voltage),
        "Current_A"      : float(current),
        "Ambient_Temp_C" : float(ambient_temp),
        "RPM"            : float(rpm),
        "Vibration_mm_s" : float(vibration),
        "Humidity_%"     : float(humidity),
        "Operating_Hours": float(operating_hours),
    }])
    X_scaled  = scaler.transform(row)          # same scaler used in training
    pred_temp = float(best_model.predict(X_scaled)[0])
    risk      = get_risk(pred_temp)

    analysis = (
        f"## Risk Assessment\n"
        f"- **Predicted Temperature**: **{pred_temp:.1f} °C**\n"
        f"- **Risk Level**: {risk}\n"
        f"- Safe: < 60 °C · Acceptable: < 80 °C · Critical: ≥ 80 °C · Danger: ≥ 100 °C\n\n"
        "---\n\n"
        "## 🔧 Scenario 1 – Preventive Maintenance\n\n"
        + scenario1_maintenance(pred_temp, float(vibration), float(operating_hours))
        + "\n\n---\n\n"
        "## ⚡ Scenario 2 – Energy Efficiency\n\n"
        + scenario2_energy(float(motor_load), pred_temp, float(voltage), float(current))
        + "\n\n---\n\n"
        "## 🏭 Scenario 3 – Equipment Reliability\n\n"
        + scenario3_reliability(pred_temp, float(motor_load),
                                float(vibration), float(operating_hours))
    )
    return f"{pred_temp:.1f} °C", risk, analysis


# ── 8. Gradio UI ─────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"),
               title="⚡ Motor Temperature Prediction") as demo:

    gr.Markdown("""
    # ⚡ Electric Motor Temperature Prediction Using Machine Learning
    ### Predictive Maintenance · Energy Efficiency · Equipment Reliability
    Set the motor parameters with the sliders, then press **Predict & Analyse**.
    """)

    with gr.Row():

        # ── Left: inputs ─────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Motor Operating Parameters")

            s_load = gr.Slider(10,    100,  value=75,   step=1,    label="Motor Load (%)",            info="% of rated full load")
            s_volt = gr.Slider(380,   440,  value=415,  step=1,    label="Voltage (V)",                info="Supply voltage")
            s_curr = gr.Slider(5,     50,   value=20,   step=0.5,  label="Current (A)",                info="Operating current")
            s_amb  = gr.Slider(15,    45,   value=25,   step=0.5,  label="Ambient Temperature (°C)",   info="Surrounding temperature")
            s_rpm  = gr.Slider(1000,  3600, value=1800, step=50,   label="Motor Speed (RPM)",          info="Rotational speed")
            s_vib  = gr.Slider(0.5,   10,   value=2.0,  step=0.1,  label="Vibration (mm/s)",           info="< 4.5 = good, > 7 = high")
            s_hum  = gr.Slider(20,    90,   value=50,   step=1,    label="Humidity (%)",               info="Relative humidity")
            s_hrs  = gr.Slider(0,     10000,value=500,  step=50,   label="Operating Hours",            info="Total cumulative run hours")

            btn_pred = gr.Button("🔍  Predict & Analyse", variant="primary", size="lg")

            gr.Markdown("**Quick Presets** – loads values into sliders and runs prediction:")
            with gr.Row():
                btn_norm = gr.Button("✅ Normal",   size="sm")
                btn_warn = gr.Button("⚠️ Warning",  size="sm")
                btn_crit = gr.Button("🚨 Critical", size="sm")

        # ── Right: outputs ────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Prediction Results")
            out_temp     = gr.Textbox(label="🌡️  Predicted Motor Temperature", interactive=False)
            out_risk     = gr.Textbox(label="⚠️   Risk Level",                  interactive=False)
            gr.Markdown("### 🎯 Scenario Analysis")
            out_analysis = gr.Markdown()

    SLIDERS = [s_load, s_volt, s_curr, s_amb, s_rpm, s_vib, s_hum, s_hrs]
    OUTPUTS = [out_temp, out_risk, out_analysis]

    # Main predict button (reads slider values)
    btn_pred.click(fn=predict, inputs=SLIDERS, outputs=OUTPUTS)

    # ── Preset functions: set sliders AND run predict ─────────
    def _run_preset(load, volt, curr, amb, rpm, vib, hum, hrs):
        t, r, a = predict(load, volt, curr, amb, rpm, vib, hum, hrs)
        # Return new slider values first, then result outputs
        return load, volt, curr, amb, rpm, vib, hum, hrs, t, r, a

    ALL_OUTPUTS = SLIDERS + OUTPUTS   # 8 sliders + 3 result widgets = 11

    btn_norm.click(fn=lambda: _run_preset(60,  415, 15, 25, 1500, 1.5, 50,  200),
                   outputs=ALL_OUTPUTS)
    btn_warn.click(fn=lambda: _run_preset(85,  420, 35, 35, 2800, 5.0, 65, 3000),
                   outputs=ALL_OUTPUTS)
    btn_crit.click(fn=lambda: _run_preset(100, 435, 48, 42, 3400, 9.0, 80, 8500),
                   outputs=ALL_OUTPUTS)

    gr.Markdown("""
    ---
    | Scenario | Purpose |
    |---|---|
    | 🔧 Preventive Maintenance | Detect overheating early, schedule inspections before failures |
    | ⚡ Energy Efficiency | Optimise motor load to reduce energy wastage and costs |
    | 🏭 Equipment Reliability | Estimate Remaining Useful Life and ensure safe operation |

    *Model trained on synthetic data. Connect a real sensor dataset for production use.*
    """)

demo.launch(share=True, debug=False)
