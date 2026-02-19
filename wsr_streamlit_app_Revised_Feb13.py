WSR_Graph_Generator.py
# Watershed Summary Report Graph Generator (Exact Style)
# Designed to take cleaned outputs from the Validation App (CSV or Excel)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, zipfile
from io import BytesIO

# ================== Streamlit Page ==================
st.set_page_config(page_title='WSR Graph Generator', layout='wide')
st.title("Watershed Summary Report Graph Generator (Exact Style)")

uploaded_file = st.file_uploader(
    "Upload your cleaned dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

# ================== Manual WQS inputs (per run) ==================

# ================== Manual WQS inputs (per run, now optional) ==================
st.sidebar.header("Water Quality Standards (Manual Input)")

segment_label = st.sidebar.text_input(
    "Segment label (for your reference)",
    value=""
)

# Helper to allow optional numeric input
def optional_number_input(label, step=1.0, default=None, help_text=""):
    val = st.sidebar.text_input(label + " (optional)", value="" if default is None else str(default), help=help_text)
    try:
        return float(val) if val.strip() != "" else None
    except:
        return None

WQS_TEMP = optional_number_input("Temperature threshold (°C)", step=0.1, default=33.9,
                                 help_text="Upper guideline temperature for this segment.")
WQS_TDS = optional_number_input("Total Dissolved Solids (mg/L)", step=10.0, default=600.0)
WQS_DO = optional_number_input("Dissolved Oxygen (mg/L) — minimum", step=0.1, default=5.0)
WQS_pH_MIN = optional_number_input("pH — minimum (s.u.)", step=0.1, default=6.5)
WQS_pH_MAX = optional_number_input("pH — maximum (s.u.)", step=0.1, default=9.0)
WQS_ECOLI_GM = optional_number_input("E. coli – Geometric Mean (#/100 mL)", step=1.0, default=126,
                                      help_text="Geometric mean criterion (e.g., 126 for PCR1).")
WQS_ECOLI_SINGLE = optional_number_input("E. coli – Single Sample (#/100 mL)", step=1.0, default=399,
                                         help_text="Single-sample maximum criterion (e.g., 399 for PCR1).")

st.sidebar.caption(
    f"Using WQS for: **{segment_label or 'Unnamed segment'}**\n\n"
    f"- Temp ≤ {WQS_TEMP if WQS_TEMP is not None else 'ND'} °C\n"
    f"- TDS ≤ {WQS_TDS if WQS_TDS is not None else 'ND'} mg/L\n"
    f"- DO ≥ {WQS_DO if WQS_DO is not None else 'ND'} mg/L\n"
    f"- pH {WQS_pH_MIN if WQS_pH_MIN is not None else 'ND'}–{WQS_pH_MAX if WQS_pH_MAX is not None else 'ND'}\n"
    f"- E. coli: {WQS_ECOLI_GM if WQS_ECOLI_GM is not None else 'ND'} GM, "
    f"{WQS_ECOLI_SINGLE if WQS_ECOLI_SINGLE is not None else 'ND'} single sample (#/100 mL)"
)



# ================== Helpers ==================
def get_col(df, *candidates):
    """Return first matching column from candidates; else NaN Series."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([np.nan] * len(df), index=df.index)

def find_first_name(df, names):
    """Return the first column name from 'names' that exists in df."""
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
    if site_order is not None and len(site_order) > 0:
        ax.set_xticks(range(1, len(site_order) + 1))
        ax.set_xticklabels(site_order, rotation=0)

def series_by_site(df, site_order, ycol):
    return [df.loc[df['Site ID'].eq(s), ycol].dropna().values for s in site_order]

# ===== Monthly Climate builder (from dataset) =====
def build_monthly_climate_from_df(df):
    date_col = find_first_name(df, ['Sample Date', 'Date', 'SampleDate', 'Datetime'])
    if date_col is None:
        return None

    temp_name = find_first_name(df, [
        'Air Temperature (° C)', 'Air Temperature (°C)',
        'Water Temperature (° C)', 'Water Temperature (°C)'
    ])
    ppt_name = find_first_name(df, [
        'Rainfall Accumulation', 'Precipitation', 'Rain', 'Rain (in)'
    ])

    tmp = df[[date_col]].copy()
    tmp['__date__'] = pd.to_datetime(tmp[date_col], errors='coerce')
    tmp = tmp.dropna(subset=['__date__']).sort_values('__date__')

    if temp_name:
        tmp['__temp__'] = to_num(df[temp_name])
    if ppt_name:
        tmp['__ppt__'] = to_num(df[ppt_name])

    has_temp = ('__temp__' in tmp.columns and tmp['__temp__'].notna().any())
    has_ppt = ('__ppt__' in tmp.columns and tmp['__ppt__'].notna().any())
    if not (has_temp or has_ppt):
        return None

    monthly = pd.DataFrame({'MonthNum': range(1, 13)})

    if has_temp:
        t_month = (
            tmp.dropna(subset=['__temp__'])
               .set_index('__date__')['__temp__']
               .resample('M').mean()
        )
        t_month = (
            t_month.reset_index()
                   .assign(MonthNum=lambda d: d['__date__'].dt.month)
                   .groupby('MonthNum', as_index=False)['__temp__'].mean()
                   .rename(columns={'__temp__': 'TempMeanC'})
        )
        monthly = monthly.merge(t_month, on='MonthNum', how='left')

    if has_ppt:
        p_month = (
            tmp.dropna(subset=['__ppt__'])
               .set_index('__date__')['__ppt__']
               .resample('M').sum()
        )
        p_month = (
            p_month.reset_index()
                   .assign(MonthNum=lambda d: d['__date__'].dt.month)
                   .groupby('MonthNum', as_index=False)['__ppt__'].sum()
                   .rename(columns={'__ppt__': 'Precip'})
        )
        monthly = monthly.merge(p_month, on='MonthNum', how='left')

    return monthly.sort_values('MonthNum')

# ================== Main ==================
if uploaded_file:
    # ---------- Read (CSV or Excel) ----------
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    st.write("Preview of input data (first 20 rows):")
    st.dataframe(df.head(20))

    # ---------- Prepare columns (mapping from cleaned file) ----------
    df['Sample Date'] = pd.to_datetime(
        get_col(df, 'Sample Date', 'Date', 'SampleDate'),
        errors='coerce'
    )

    df['Site ID'] = (
        get_col(df, 'Site ID: Site Name', 'Site ID', 'Station ID')
        .astype(str)
        .str.strip()
    )


    # ================== UNIQUE SITE ID HANDLING ==================
    desc_col = find_first_name(
        df,
        ['Description', 'Site Description', 'Site Desc', 'Location', 'Description ']
    )

    # Temperatures
    df['Air Temp Rounded'] = to_num(
        get_col(df,
                'Air Temp Rounded',
                'Air Temperature (° C)',
                'Air Temperature (°C)')
    )
    df['Water Temp Rounded'] = to_num(
        get_col(df,
                'Water Temp Rounded',
                'Water Temperature (° C)',
                'Water Temperature (°C)')
    )

    # Conductivity
    cond_col = get_col(
        df,
        'Conductivity (µS/cm)', 'Conductivity (uS/cm)',
        'Conductivity (?S/cm)', 'Conductivity',
        'Conductivity (μS/cm)'
    )
    df['Conductivity'] = to_num(cond_col)

    # TDS: use existing if present, else compute from conductivity
    tds_existing = get_col(df, 'TDS (mg/L)', 'Total Dissolved Solids (mg/L)')
    if tds_existing.notna().sum() > 0:
        df['TDS (mg/L)'] = to_num(tds_existing)
    else:
        df['TDS (mg/L)'] = df['Conductivity'] * 0.65

    # DO
    df['DO_avg'] = to_num(
        get_col(df,
                'Dissolved Oxygen (mg/L) Average',
                'DO_avg',
                'Dissolved Oxygen (mg/L)')
    )
    # pH
    df['pH'] = to_num(
        get_col(df,
                'pH Rounded',
                'pH (standard units)',
                'pH')
    )

    # Transparency, depth
    df['Secchi'] = to_num(
        get_col(df,
                'Secchi Disk Transparency - Average',
                'Secchi Disk Transparency (m)',
                'Secchi')
    )
    df['Transparency Tube'] = to_num(
        get_col(df,
                'Transparency Tube (meters)',
                'Transparency Tube (m)')
    )
    df['Total Depth'] = to_num(
        get_col(df,
                'Total Depth (meters)',
                'Total Depth (m)',
                'Depth (m)')
    )

    # E. coli
    ecoli_col = get_col(
        df,
        'E. coli (CFU/100 mL)',
        'E. coli (#/100 mL)',
        'E. coli',
        'E.coli (CFU/100 mL)',
        'E Coli (CFU/100 mL)',
        'E_coli',  # from cleaned CSV
        'E. Coli Average'
    )
    df['E_coli'] = to_num(ecoli_col)

    # Drop rows without Site ID
    df = df[df['Site ID'].notna() & df['Site ID'].ne('')].copy()

    if df.empty:
        st.error("No valid Site ID values found after cleaning. Please check your input file.")
        st.stop()

    # ---------- Output directory ----------
    output_dir = "wsr_figures"
    ensure_dir(output_dir)

    # Keep site order as in file
    site_order = list(pd.unique(df['Site ID']))

    # ================== Figure 6: Water Temperature ==================
    fig6, ax = plt.subplots(figsize=(14, 6))
    for s in site_order:
        dsi = df[df['Site ID'].eq(s)]
        ax.scatter(
            dsi['Sample Date'],
            dsi['Water Temp Rounded'],
            s=40,
            marker='o',
            label=s,
            alpha=0.9
        )
    ax.axhline(WQS_TEMP, linestyle='--', color='red', linewidth=1.5, zorder=10)
    if df['Sample Date'].notna().any():
        xmin = df['Sample Date'].min()
        ax.text(xmin, WQS_TEMP + 0.5, 'WQS', color='red', va='bottom', zorder=11)
    ax.set_xlabel('Sample Date')
    ax.set_ylabel('Water Temperature (°C)')
    ax.set_title(f"{segment_label}")
    ax.legend(title='Site ID', loc='center left', bbox_to_anchor=(1.0, 0.5))
    save_figure(fig6, os.path.join(output_dir, "Figure6_WaterTemperature.png"))

    # ================== Figure 7: TDS ==================
    fig7, ax = plt.subplots(figsize=(10, 6))
    tds_by_site = series_by_site(df, site_order, 'TDS (mg/L)')
    ax.boxplot(
        tds_by_site,
        patch_artist=False, whis=1.5,
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        boxprops=dict(color='black'),
        flierprops=dict(
            marker='o', markersize=4,
            markerfacecolor='black', markeredgecolor='black'
        )
    )
    style_axes(ax, 'Site ID', 'Total Dissolved Solids (mg/L)', site_order)

    # Make sure WQS line is visible
    if any(len(v) > 0 for v in tds_by_site):
        all_vals = np.concatenate([v for v in tds_by_site if len(v) > 0])
        y_min = float(np.nanmin(all_vals))
        y_max = float(np.nanmax(all_vals))
        span = (y_max - y_min) if (y_max > y_min) else 10.0
        pad = max(5.0, 0.05 * span)
        y_min_adj = min(y_min - pad, WQS_TDS - pad)
        y_max_adj = max(y_max + pad, WQS_TDS + pad)
        ax.set_ylim(y_min_adj, y_max_adj)
    else:
        ax.set_ylim(WQS_TDS - 50, WQS_TDS + 50)

    ax.axhline(WQS_TDS, linestyle='--', color='red', linewidth=1.8, zorder=10)
    ax.text(
        1,
        WQS_TDS + (0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
        'WQS', color='red', va='bottom', zorder=11
    )
    ax.set_title(f"{segment_label}")
    save_figure(fig7, os.path.join(output_dir, "Figure7_TDS_Boxplot.png"))

    # ================== Figure 7B: Conductivity ==================
    fig7b, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        series_by_site(df, site_order, 'Conductivity'),
        patch_artist=False,
        whis=1.5,
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        boxprops=dict(color='black'),
        flierprops=dict(
            marker='o',
            markersize=4,
            markerfacecolor='black',
            markeredgecolor='black'
        )
    )
    style_axes(ax, 'Site ID', 'Specific Conductance (µS/cm)', site_order)
    ax.set_title(f"{segment_label}")
    save_figure(fig7b, os.path.join(output_dir, "Figure7B_Conductivity_Boxplot.png"))

    # ================== Figure 8: Dissolved Oxygen ==================
    fig8, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        series_by_site(df, site_order, 'DO_avg'),
        patch_artist=False, whis=1.5,
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        boxprops=dict(color='black'),
        flierprops=dict(
            marker='o', markersize=4,
            markerfacecolor='black', markeredgecolor='black'
        )
    )
    style_axes(ax, 'Site ID', 'Dissolved Oxygen (mg/L)', site_order)
    ax.axhline(WQS_DO, linestyle='--', color='red', zorder=10)
    ax.text(1, WQS_DO + 0.1, 'WQS', color='red', va='bottom', zorder=11)
    ax.set_title(f"{segment_label}")
    save_figure(fig8, os.path.join(output_dir, "Figure8_DO_Boxplot.png"))

    # ================== Figure 9: pH ==================
    fig_ph, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        series_by_site(df, site_order, 'pH'),
        patch_artist=False, whis=1.5,
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        boxprops=dict(color='black'),
        flierprops=dict(
            marker='o', markersize=4,
            markerfacecolor='black', markeredgecolor='black'
        )
    )
    style_axes(ax, 'Site ID', 'pH (standard units)', site_order)
    ax.axhline(WQS_pH_MAX, linestyle='--', color='red', zorder=10)
    ax.axhline(WQS_pH_MIN, linestyle='--', color='red', zorder=10)
    ax.text(1, WQS_pH_MAX + 0.03, 'WQS Max', color='red', va='bottom', zorder=11)
    ax.text(1, WQS_pH_MIN + 0.03, 'WQS Min', color='red', va='bottom', zorder=11)
    ax.set_title(f"{segment_label}")
    save_figure(fig_ph, os.path.join(output_dir, "Figure9_pH_Boxplot.png"))

    # ================== Figure 10: Transparency (Secchi vs Tube) ==================
    trans_df = df.melt(
        id_vars=['Site ID'],
        value_vars=['Secchi', 'Transparency Tube'],
        var_name='Type', value_name='Value'
    ).dropna()

    fig10, ax = plt.subplots(figsize=(12, 6))
    pos = np.arange(1, len(site_order) + 1)
    offset = 0.18
    type2shift = {'Secchi': -offset, 'Transparency Tube': +offset}
    type2color = {'Secchi': 'blue', 'Transparency Tube': 'red'}

    for t in ['Secchi', 'Transparency Tube']:
        data_t = [
            trans_df[
                (trans_df['Site ID'].eq(s)) & (trans_df['Type'].eq(t))
            ]['Value'].dropna().values
            for s in site_order
        ]
        ax.boxplot(
            data_t,
            positions=pos + type2shift[t],
            widths=0.28,
            patch_artist=True, whis=1.5,
            medianprops=dict(color='black'),
            whiskerprops=dict(color=type2color[t]),
            capprops=dict(color=type2color[t]),
            boxprops=dict(facecolor='white', edgecolor=type2color[t]),
            flierprops=dict(
                marker='o', markersize=4,
                markerfacecolor=type2color[t], markeredgecolor='black'
            )
        )

    style_axes(ax, 'Site ID', 'Transparency (m)', site_order)
    ax.set_ylim(0, 1.4)
    handles = [
        plt.Line2D([0], [0], color='blue', lw=2, label='Secchi Disk'),
        plt.Line2D([0], [0], color='red', lw=2, label='Transparency Tube')
    ]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_title(f"{segment_label}")
    save_figure(fig10, os.path.join(output_dir, "Figure10_Transparency_Boxplot.png"))

    # ================== Figure 10B: Transparency Tube ONLY ==================
    fig10b = None
    if 'Transparency Tube' in df.columns and df['Transparency Tube'].notna().any():
        fig10b, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(
            series_by_site(df, site_order, 'Transparency Tube'),
            patch_artist=False, whis=1.5,
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            boxprops=dict(color='black'),
            flierprops=dict(
                marker='o', markersize=4,
                markerfacecolor='black', markeredgecolor='black'
            )
        )
        style_axes(ax, 'Site ID', 'Transparency Tube (m)', site_order)
        ax.set_ylim(0, 1.4)
        ax.set_title(f"{segment_label}")
        save_figure(fig10b, os.path.join(output_dir, "Figure10B_TransparencyTube_Boxplot.png"))

    # ================== Figure 11: Total Depth ==================
    fig11, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        series_by_site(df, site_order, 'Total Depth'),
        patch_artist=False, whis=1.5,
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        boxprops=dict(color='black'),
        flierprops=dict(
            marker='o', markersize=4,
            markerfacecolor='black', markeredgecolor='black'
        )
    )
    style_axes(ax, 'Site ID', 'Total Depth (m)', site_order)
    ax.set_title(f"{segment_label}")
    save_figure(fig11, os.path.join(output_dir, "Figure11_TotalDepth_Boxplot.png"))

    # ================== Figure 12: E. coli ==================
    fig12 = None
    if 'E_coli' in df.columns and df['E_coli'].notna().any():
        fig12, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(
            series_by_site(df, site_order, 'E_coli'),
            patch_artist=False, whis=1.5,
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            boxprops=dict(color='black'),
            flierprops=dict(
                marker='o',
                markersize=4,
                markerfacecolor='black',
                markeredgecolor='black'
            )
        )

        style_axes(ax, 'Site ID', 'E. coli (CFU/100 mL)', site_order)

        # Geometric mean WQS line
        ax.axhline(WQS_ECOLI_GM, linestyle='--', color='red', zorder=10)
        ax.text(
            1, WQS_ECOLI_GM * 1.02,
            'GM WQS', color='red', va='bottom', zorder=11
        )

        # Single-sample WQS line
        ax.axhline(WQS_ECOLI_SINGLE, linestyle=':', color='red', zorder=10)
        ax.text(
            1, WQS_ECOLI_SINGLE * 1.02,
            'Single-sample WQS', color='red', va='bottom', zorder=11
        )

        ax.set_title(f"{segment_label}")
        save_figure(fig12, os.path.join(output_dir, "Figure12_Ecoli_Boxplot.png"))

    # ================== Monthly Climate ==================
    fig_climate = None
    monthly_climate = build_monthly_climate_from_df(df)
    if monthly_climate is not None:
        month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec']
        monthly_climate['Month'] = monthly_climate['MonthNum'].map(
            {i+1: m for i, m in enumerate(month_labels)}
        )

        fig_climate, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Month')

        if 'TempMeanC' in monthly_climate.columns:
            ax1.set_ylabel('Temperature (°C)', color='red')
            ax1.plot(
                monthly_climate['Month'],
                monthly_climate['TempMeanC'],
                color='red', linewidth=3
            )
            ax1.tick_params(axis='y', labelcolor='red')

        ax2 = ax1.twinx()
        if 'Precip' in monthly_climate.columns:
            ax2.set_ylabel('Precipitation', color='blue')
            ax2.bar(
                monthly_climate['Month'],
                monthly_climate['Precip'],
                alpha=0.7, color='blue'
            )
            ax2.tick_params(axis='y', labelcolor='blue')

        fig_climate.tight_layout()
        save_figure(fig_climate, os.path.join(output_dir, "Figure_MonthlyClimate.png"))
    else:
        st.warning("Monthly climate could not be built (no usable date/temperature/precip columns).")

    # ================== Summary Table (Table 6 style) ==================
    param_map = {
        'Air Temp Rounded': 'Air Temperature (°C)',
        'Water Temp Rounded': 'Water Temperature (°C)',
        'DO_avg': 'Dissolved Oxygen (mg/L)',
        'pH': 'pH (standard units)',
        'Conductivity': 'Conductivity (µS/cm)',
        'Secchi': 'Secchi Disk Transparency (m)',
        'Transparency Tube': 'Transparency Tube (m)',
        'Total Depth': 'Total Depth (m)',
        'TDS (mg/L)': 'Total Dissolved Solids (mg/L)',
        'E_coli': 'E. coli (CFU/100 mL)',
    }

    summary_rows = []

    for col, pname in param_map.items():
        if col in df.columns and df[col].notna().any():

            # Different stats for E. coli
            if col == 'E_coli':
                stats = ['Geometric Mean', 'Range']
            else:
                stats = ['Mean', 'Std Dev', 'Range']

            for stat in stats:
                row = {'Parameter': pname, 'Statistic': stat}

                for s in site_order:
                    vals = df.loc[df['Site ID'].eq(s), col].dropna().values

                    if len(vals) == 0:
                        row[s] = np.nan
                        continue

                    if col == 'E_coli' and stat == 'Geometric Mean':
                        # Avoid log(0) by adding offset
                        vals_adj = np.where(vals <= 0, 0.5, vals)
                        gm = float(np.exp(np.mean(np.log(vals_adj))))
                        row[s] = gm

                    elif stat == 'Mean':
                        row[s] = float(np.mean(vals))

                    elif stat == 'Std Dev':
                        row[s] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

                    elif stat == 'Range':
                        row[s] = float(np.max(vals) - np.min(vals))

                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    value_cols = [c for c in summary_df.columns if c not in ['Parameter', 'Statistic']]
    for c in value_cols:
        summary_df[c] = pd.to_numeric(summary_df[c], errors='coerce')
    summary_df[value_cols] = summary_df[value_cols].round(2)

    st.subheader("Summary Statistics (Table 6 style)")
    if not summary_df.empty:
        st.dataframe(
            summary_df.style.format(
                {c: "{:.2f}" for c in value_cols},
                na_rep="ND"
            )
        )
    else:
        st.info("No numeric data found to summarize for Table 6.")

    table6_path = os.path.join(output_dir, "Table6_Summary.xlsx")
    if not summary_df.empty:
        save_df = summary_df.copy()
        save_df[value_cols] = save_df[value_cols].applymap(
            lambda v: v if pd.notna(v) else "ND"
        )
        save_df.to_excel(table6_path, index=False)

    # ================== ZIP download ==================
    st.markdown("## Download All Results (Figures + Table 6)")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for f in os.listdir(output_dir):
            zipf.write(os.path.join(output_dir, f), arcname=f)
    zip_buffer.seek(0)
    st.download_button(
        "Download ZIP",
        data=zip_buffer,
        file_name="WSR_All_Results.zip",
        mime="application/zip"
    )

    # ================== Show charts ==================
    st.subheader("Figure 6. Water Temperature Over Time by Site")
    st.pyplot(fig6); plt.close(fig6)

    st.subheader("Figure 7. TDS by Site")
    st.pyplot(fig7); plt.close(fig7)

    st.subheader("Figure 7B. Specific Conductance by Site")
    st.pyplot(fig7b); plt.close(fig7b)

    st.subheader("Figure 8. Dissolved Oxygen by Site")
    st.pyplot(fig8); plt.close(fig8)

    st.subheader("Figure 9. pH by Site")
    st.pyplot(fig_ph); plt.close(fig_ph)

    st.subheader("Figure 10. Transparency by Site (Secchi vs Tube)")
    st.pyplot(fig10); plt.close(fig10)

    if fig10b is not None:
        st.subheader("Figure 10B. Transparency Tube by Site")
        st.pyplot(fig10b); plt.close(fig10b)

    st.subheader("Figure 11. Total Depth by Site")
    st.pyplot(fig11); plt.close(fig11)

    if fig12 is not None:
        st.subheader("Figure 12. E. coli by Site")
        st.pyplot(fig12); plt.close(fig12)

    if fig_climate is not None:
        st.subheader("Monthly Average Temperature and Total Precipitation")
        st.pyplot(fig_climate); plt.close(fig_climate)

