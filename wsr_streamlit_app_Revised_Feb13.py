# WSR_Graph_Generator_Optional.py
# Watershed Summary Report Graph Generator (Optional WQS & Graphs)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, zipfile
from io import BytesIO

# ================== Streamlit Page ==================
st.set_page_config(page_title='WSR Graph Generator', layout='wide')
st.title("Watershed Summary Report Graph Generator (Optional WQS & Graphs)")

uploaded_file = st.file_uploader(
    "Upload your cleaned dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

# ================== Manual WQS inputs (optional) ==================
st.sidebar.header("Water Quality Standards (Manual Input — optional)")
segment_label = st.sidebar.text_input("Segment label (for your reference)", value="")

def optional_number_input(label, step=1.0, fmt="%.1f", default=None, help_text=""):
    val = st.sidebar.text_input(label + " (optional)", value="" if default is None else str(default), help=help_text)
    try:
        return float(val) if val.strip() != "" else None
    except:
        return None

WQS_TEMP = optional_number_input("Temperature threshold (°C)", step=0.1, fmt="%.1f", default=33.9, help_text="Upper guideline temperature")
WQS_TDS = optional_number_input("Total Dissolved Solids (mg/L)", step=10.0, fmt="%.0f", default=600, help_text="TDS guideline")
WQS_DO = optional_number_input("Dissolved Oxygen (mg/L) — minimum", step=0.1, fmt="%.1f", default=5.0)
WQS_pH_MIN = optional_number_input("pH — minimum (s.u.)", step=0.1, fmt="%.1f", default=6.5)
WQS_pH_MAX = optional_number_input("pH — maximum (s.u.)", step=0.1, fmt="%.1f", default=9.0)
WQS_ECOLI_GM = optional_number_input("E. coli – Geometric Mean (#/100 mL)", step=1.0, fmt="%.0f", default=126)
WQS_ECOLI_SINGLE = optional_number_input("E. coli – Single Sample (#/100 mL)", step=1.0, fmt="%.0f", default=399)

# ================== Helpers ==================
def get_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([np.nan]*len(df), index=df.index)

def find_first_name(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def to_num(s):
    return pd.to_numeric(s, errors='coerce')

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight')

def style_axes(ax, xlabel='', ylabel='', site_order=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_color('black')
    if site_order:
        ax.set_xticks(range(1, len(site_order)+1))
        ax.set_xticklabels(site_order, rotation=0)

def series_by_site(df, site_order, ycol):
    return [df.loc[df['Site ID'].eq(s), ycol].dropna().values for s in site_order]

def build_monthly_climate_from_df(df):
    date_col = find_first_name(df, ['Sample Date', 'Date', 'SampleDate', 'Datetime'])
    if date_col is None:
        return None

    temp_name = find_first_name(df, ['Air Temperature (° C)', 'Air Temperature (°C)',
                                     'Water Temperature (° C)', 'Water Temperature (°C)'])
    ppt_name = find_first_name(df, ['Rainfall Accumulation', 'Precipitation', 'Rain', 'Rain (in)'])

    tmp = df[[date_col]].copy()
    tmp['__date__'] = pd.to_datetime(tmp[date_col], errors='coerce')
    tmp = tmp.dropna(subset=['__date__']).sort_values('__date__')

    if temp_name: tmp['__temp__'] = to_num(df[temp_name])
    if ppt_name: tmp['__ppt__'] = to_num(df[ppt_name])

    has_temp = '__temp__' in tmp.columns and tmp['__temp__'].notna().any()
    has_ppt = '__ppt__' in tmp.columns and tmp['__ppt__'].notna().any()
    if not (has_temp or has_ppt):
        return None

    monthly = pd.DataFrame({'MonthNum': range(1, 13)})

    if has_temp:
        t_month = tmp.dropna(subset=['__temp__']).set_index('__date__')['__temp__'].resample('M').mean()
        t_month = t_month.reset_index().assign(MonthNum=lambda d: d['__date__'].dt.month)\
            .groupby('MonthNum', as_index=False)['__temp__'].mean().rename(columns={'__temp__': 'TempMeanC'})
        monthly = monthly.merge(t_month, on='MonthNum', how='left')

    if has_ppt:
        p_month = tmp.dropna(subset=['__ppt__']).set_index('__date__')['__ppt__'].resample('M').sum()
        p_month = p_month.reset_index().assign(MonthNum=lambda d: d['__date__'].dt.month)\
            .groupby('MonthNum', as_index=False)['__ppt__'].sum().rename(columns={'__ppt__': 'Precip'})
        monthly = monthly.merge(p_month, on='MonthNum', how='left')

    return monthly.sort_values('MonthNum')

# ================== Main ==================
if uploaded_file:
    file_name = uploaded_file.name.lower()
    df = pd.read_csv(uploaded_file) if file_name.endswith(".csv") else pd.read_excel(uploaded_file)

    st.success(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(df.head(20))

    # Standardize key columns
    df['Sample Date'] = pd.to_datetime(get_col(df, 'Sample Date', 'Date', 'SampleDate'), errors='coerce')
    df['Site ID'] = get_col(df, 'Site ID: Site Name', 'Site ID', 'Station ID').astype(str).str.strip()
    df = df[df['Site ID'].notna() & df['Site ID'].ne('')].copy()
    site_order = list(pd.unique(df['Site ID']))

    # Columns mapping for optional plotting
    param_map = {
        'Water Temp Rounded': ('Water Temperature (°C)', WQS_TEMP),
        'TDS (mg/L)': ('Total Dissolved Solids (mg/L)', WQS_TDS),
        'Conductivity': ('Specific Conductance (µS/cm)', None),
        'DO_avg': ('Dissolved Oxygen (mg/L)', WQS_DO),
        'pH': ('pH (standard units)', (WQS_pH_MIN, WQS_pH_MAX)),
        'Secchi': ('Secchi Disk Transparency (m)', None),
        'Transparency Tube': ('Transparency Tube (m)', None),
        'Total Depth': ('Total Depth (m)', None),
        'E_coli': ('E. coli (CFU/100 mL)', (WQS_ECOLI_GM, WQS_ECOLI_SINGLE)),
    }

    output_dir = "wsr_figures"
    ensure_dir(output_dir)

    # ---------------- Plotting ----------------
    for col, (label, wqs) in param_map.items():
        if col not in df.columns or df[col].notna().sum() == 0:
            st.info(f"Skipping {label} graph: no data.")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        if col == 'Water Temp Rounded':
            for s in site_order:
                dsi = df[df['Site ID'].eq(s)]
                ax.scatter(dsi['Sample Date'], dsi[col], s=40, label=s, alpha=0.9)
            if wqs is not None:
                ax.axhline(wqs, linestyle='--', color='red', linewidth=1.5, zorder=10)
        else:
            data_by_site = series_by_site(df, site_order, col)
            ax.boxplot(
                data_by_site,
                patch_artist=False, whis=1.5,
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                boxprops=dict(color='black'),
                flierprops=dict(marker='o', markersize=4, markerfacecolor='black', markeredgecolor='black')
            )
            if isinstance(wqs, (tuple, list)) and len(wqs) == 2:  # min/max
                if wqs[0] is not None: ax.axhline(wqs[0], linestyle='--', color='red')
                if wqs[1] is not None: ax.axhline(wqs[1], linestyle='--', color='red')
            elif wqs is not None:
                ax.axhline(wqs, linestyle='--', color='red')

        style_axes(ax, 'Site ID', label, site_order)
        ax.set_title(f"{segment_label} — {label}")
        save_figure(fig, os.path.join(output_dir, f"{col}_plot.png"))
        st.pyplot(fig); plt.close(fig)

    # ---------------- Monthly Climate ----------------
    monthly_climate = build_monthly_climate_from_df(df)
    if monthly_climate is not None:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        monthly_climate['Month'] = monthly_climate['MonthNum'].map({i+1: m for i, m in enumerate(month_labels)})
        if 'TempMeanC' in monthly_climate.columns:
            ax1.plot(monthly_climate['Month'], monthly_climate['TempMeanC'], color='red', linewidth=3)
            ax1.set_ylabel('Temperature (°C)', color='red')
        ax2 = ax1.twinx()
        if 'Precip' in monthly_climate.columns:
            ax2.bar(monthly_climate['Month'], monthly_climate['Precip'], alpha=0.7, color='blue')
            ax2.set_ylabel('Precipitation', color='blue')
        fig.tight_layout()
        save_figure(fig, os.path.join(output_dir, "MonthlyClimate.png"))
        st.pyplot(fig); plt.close(fig)
    else:
        st.warning("Skipping Monthly Climate: no usable date/temp/precip data.")

    # ---------------- ZIP download ----------------
    st.markdown("## Download All Results (Figures)")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for f in os.listdir(output_dir):
            zipf.write(os.path.join(output_dir, f), arcname=f)
    zip_buffer.seek(0)
    st.download_button(
        "Download ZIP",
        data=zip_buffer,
        file_name="WSR_Optional_Results.zip",
        mime="application/zip"
    )
