import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

st.set_page_config(page_title="Chromatogram Deconvolution", layout="wide")

st.title("Chromatographic Signal Deconvolution")
st.write("Upload an Excel file with time in the first column and RI intensity in the second column.")

# -----------------------------
# Gaussian functions
# -----------------------------
def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

def two_gaussians(x, a1, c1, s1, a2, c2, s2, baseline):
    return (
        gaussian(x, a1, c1, s1)
        + gaussian(x, a2, c2, s2)
        + baseline
    )

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    time = df.iloc[:, 0].to_numpy(dtype=float)
    signal = df.iloc[:, 1].to_numpy(dtype=float)

    mask = np.isfinite(time) & np.isfinite(signal)
    time = time[mask]
    signal = signal[mask]

    baseline_guess = np.min(signal)
    signal_corrected = signal - baseline_guess

    max_signal = np.max(signal_corrected)
    time_min = np.min(time)
    time_max = np.max(time)
    time_range = time_max - time_min

    initial_guess = [
        max_signal, time_min + time_range * 0.35, time_range * 0.05,
        max_signal / 2, time_min + time_range * 0.65, time_range * 0.05,
        baseline_guess
    ]

    lower_bounds = [
        0, time_min, 0,
        0, time_min, 0,
        -np.inf
    ]

    upper_bounds = [
        np.inf, time_max, time_range,
        np.inf, time_max, time_range,
        np.inf
    ]

    try:
        params, covariance = curve_fit(
            two_gaussians,
            time,
            signal,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000
        )

        a1, c1, s1, a2, c2, s2, baseline = params

        peak1 = gaussian(time, a1, c1, s1)
        peak2 = gaussian(time, a2, c2, s2)
        fit_total = two_gaussians(time, *params)

        area1 = a1 * s1 * np.sqrt(2 * np.pi)
        area2 = a2 * s2 * np.sqrt(2 * np.pi)

        peak1_y = gaussian(c1, a1, c1, s1) + baseline
        peak2_y = gaussian(c2, a2, c2, s2) + baseline

        results = pd.DataFrame({
            "Peak": ["Peak 1", "Peak 2"],
            "Peak position / maximum time": [c1, c2],
            "Standard deviation": [s1, s2],
            "Amplitude": [a1, a2],
            "Area": [area1, area2]
        })

        st.subheader("Fit results")
        st.dataframe(results)

        st.write(f"**Baseline:** {baseline:.4f}")

        # -----------------------------
        # Plot
        # -----------------------------
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(time, signal, "k.", label="Raw RI signal", markersize=4)
        ax.plot(time, fit_total, "r-", label="Total fit", linewidth=2)
        ax.plot(time, peak1 + baseline, "--", label="Gaussian peak 1")
        ax.plot(time, peak2 + baseline, "--", label="Gaussian peak 2")

        ax.plot(c1, peak1_y, "o", markersize=8)
        ax.plot(c2, peak2_y, "o", markersize=8)

        ax.text(
            c1,
            peak1_y,
            f"Peak 1\nposition={c1:.2f}\nstd={s1:.2f}",
            ha="left",
            va="bottom"
        )

        ax.text(
            c2,
            peak2_y,
            f"Peak 2\nposition={c2:.2f}\nstd={s2:.2f}",
            ha="left",
            va="bottom"
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("RI intensity")
        ax.set_title("Chromatographic Signal Deconvolution: 2 Gaussian Peaks")
        ax.legend()

        st.subheader("Fit figure")
        st.pyplot(fig)

        # -----------------------------
        # Download results as CSV
        # -----------------------------
        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="deconvolution_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("The fit failed. Try checking the Excel file or adjusting the data range.")
        st.write(e)
else:
    st.info("Please upload an Excel file to start.")