import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# -------------------- Page Config --------------------

st.set_page_config(
    page_title="Dashboard Izin Terintegrasi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Konstanta --------------------
REQUIRED_COLS = ["SEKTOR", "JENIS IZIN", "KATEGORI", "BULAN", "JUMLAH"]
MONTH_ORDER_ID = [
    "JANUARI", "FEBRUARI", "MARET", "APRIL", "MEI", "JUNI",
    "JULI", "AGUSTUS", "SEPTEMBER", "OKTOBER", "NOVEMBER", "DESEMBER"
]

# -------------------- Fungsi Preprocessing --------------------
@st.cache_data(show_spinner=True)
def preprocess_raw(file):
    df = pd.read_excel(file, header=None, dtype=object)

    # Step sesuai preprocesing.py
    if df.shape[0] >= 5:
        df = df.drop(index=4).reset_index(drop=True)
    to_drop = []
    if df.shape[0] >= 1:
        to_drop.append(0)
    if df.shape[0] >= 76:
        to_drop.append(75)
    if to_drop:
        df = df.drop(index=to_drop).reset_index(drop=True)

    if df.shape[0] >= 1:
        df = df.drop(index=0).reset_index(drop=True)

    df = df[df.index < 72].reset_index(drop=True)
    df = df.T
    df = df[df.index < 56].reset_index(drop=True)
    df.iloc[:, 0] = df.iloc[:, 0].ffill()
    df.iloc[:, 0] = df.iloc[:, 0].astype(str) + ',' + df.iloc[:, 1].astype(str)
    df = df.drop(df.columns[1], axis=1)
    df = df.T.reset_index(drop=True)
    df.columns = df.iloc[0]
    df = df.drop(0).reset_index(drop=True)

    # drop kolom tidak perlu
    cols_to_drop = [
        col for col in df.columns
        if isinstance(col, str) and (
            "JUMLAH DI FO" in col.upper() or
            "KETERANGAN" in col.upper() or
            col.upper().endswith("JUMLAH")
        )
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df = df.fillna(0).infer_objects(copy=False)

    # melt
    id_vars = [col for col in df.columns if str(col).startswith(("NO", "SEKTOR", "JENIS IZIN"))]
    df_melted = df.melt(id_vars=id_vars, var_name="KATEGORI", value_name="JUMLAH")
    df_melted[['KATEGORI', 'BULAN']] = df_melted['KATEGORI'].str.split(',', expand=True)

    # urutkan & tipe data
    df_melted["BULAN"] = df_melted["BULAN"].astype(str).str.upper().str.strip()
    df_melted["BULAN"] = pd.Categorical(df_melted["BULAN"], categories=MONTH_ORDER_ID, ordered=True)
    df_melted = df_melted.sort_values("BULAN")
    df_melted["JUMLAH"] = pd.to_numeric(df_melted["JUMLAH"], errors="coerce").fillna(0).astype(int)
    df_melted.columns = df_melted.columns.map(lambda x: str(x).replace(",nan", "").strip())


    return df_melted

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    uploaded = st.file_uploader("Unggah file Excel mentah", type=["xlsx"], key="file_uploader")

if uploaded:
    df = preprocess_raw(uploaded)

    with st.sidebar:
        st.success(f"✅ Data berhasil diproses ({len(df)} baris).")
        sektor_opts = sorted(df["SEKTOR"].dropna().unique().tolist())
        jenis_opts  = sorted(df["JENIS IZIN"].dropna().unique().tolist())
        bulan_opts  = [m for m in MONTH_ORDER_ID if m in df["BULAN"].unique().tolist()]

        with st.expander("Sektor", expanded=True):
            all_sektor = st.checkbox("Pilih Semua Sektor", value=True)
            sel_sektor = st.multiselect("Cari / pilih sektor", sektor_opts, default=(sektor_opts if all_sektor else []))

        with st.expander("Jenis Izin", expanded=True):
            all_jenis = st.checkbox("Pilih Semua Jenis Izin", value=True)
            sel_jenis = st.multiselect("Cari / pilih jenis izin", jenis_opts, default=(jenis_opts if all_jenis else []))

        with st.expander("Bulan", expanded=True):
            all_bulan = st.checkbox("Pilih Semua Bulan", value=True)
            sel_bulan = st.multiselect("Cari / pilih bulan", bulan_opts, default=(bulan_opts if all_bulan else []))

        min_jml, max_jml = int(df["JUMLAH"].min()), int(df["JUMLAH"].max())
        rng = st.slider("Rentang Jumlah", min_value=min_jml, max_value=max_jml, value=(min_jml, max_jml))

        top_n = st.number_input("Top-N Jenis Izin untuk Tren", min_value=1, max_value=20, value=5, step=1)

    # -------------------- Main Page --------------------
    st.title("📊 Dashboard Izin Multi-Kategori")
    kategori_order = ["JUMLAH IZIN", "JUMLAH BERKAS DICABUT", "JUMLAH PENOLAKAN", "JUMLAH IZIN TERBIT"]
    kategori_opts = [k for k in kategori_order if k in df["KATEGORI"].unique()]
    kategori_pilih = st.radio("Pilih Kategori", kategori_opts, horizontal=True)

    dff = df[
        (df["KATEGORI"] == kategori_pilih) &
        (df["SEKTOR"].isin(sel_sektor)) &
        (df["JENIS IZIN"].isin(sel_jenis)) &
        (df["BULAN"].isin(sel_bulan)) &
        (df["JUMLAH"].between(rng[0], rng[1]))
    ].copy()

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total", f"{int(dff['JUMLAH'].sum()):,}".replace(",", "."))
    with c2: st.metric("Jumlah Sektor", dff["SEKTOR"].nunique())
    top_izin = dff.groupby("JENIS IZIN")["JUMLAH"].sum().sort_values(ascending=False).reset_index()
    izin_top = top_izin.iloc[0] if not top_izin.empty else None
    delta_izin = f"Top: {izin_top['JENIS IZIN']}" if izin_top is not None else "-"
    with c3:
        st.metric("Jenis Izin", dff["JENIS IZIN"].nunique(), delta=delta_izin)

    if not dff.empty:
        bulan_group = dff.groupby("BULAN", as_index=False)["JUMLAH"].sum()
        bulan_peak = bulan_group.loc[bulan_group["JUMLAH"].idxmax()]

        total_all = int(dff["JUMLAH"].sum())
        bulan_val = int(bulan_peak["JUMLAH"])
        persen_peak = (bulan_val / total_all * 100) if total_all > 0 else 0

        with c4:
         st.metric(
            "Puncak Bulan",
            str(bulan_peak["BULAN"]),
            delta=f"{bulan_val:,} izin ({persen_peak:.1f}%)".replace(",", ".")
        )
    else:
        with c4:
            st.metric("Puncak Bulan", "-", delta="-")
    st.divider()

    # Charts
    # -------------------- CHARTS --------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Total per Bulan", "Sektor per Bulan", "Tren Top-N Jenis",
        "Komposisi Bulan", "Perbandingan Kategori", "Tabel & Unduh"
    ])

    # === Tab 1: Total per Bulan ===
    # === Tab 1: Total per Bulan ===
    with tab1:
        # --- Data agregasi ---
        g1 = dff.groupby("BULAN", as_index=False)["JUMLAH"].sum()
        kategori_summary = (
            df.groupby("KATEGORI", as_index=False)["JUMLAH"]
            .sum()
            .sort_values("JUMLAH", ascending=False)
        )
        total_all = kategori_summary["JUMLAH"].sum()
        kategori_summary["PERSENTASE"] = (kategori_summary["JUMLAH"] / total_all * 100).round(1)

        # --- Header sejajar ---
        header_col1, header_col2 = st.columns([2, 1])
        with header_col1:
            st.subheader("Total per Bulan")
        with header_col2:
            st.subheader("Komposisi Kategori")

        # --- Chart sejajar ---
        col1, col2 = st.columns([2, 1])

        with col1:
            g1_nonzero = g1[g1["JUMLAH"] > 0]
            fig1 = px.bar(
                g1_nonzero, x="BULAN", y="JUMLAH", text="JUMLAH",
                color_discrete_sequence=["#FF4B4B"]
            )
            fig1.update_traces(texttemplate='%{text}', textposition='outside')
            fig1.update_layout(
                yaxis_title="Jumlah", 
                xaxis_title="Bulan",
                height=550,
                margin=dict(t=20, b=20, l=10, r=10)
            )
            st.plotly_chart(fig1, use_container_width=True, key="fig_total_per_bulan")

        with col2:
            # --- Donut chart ---
            colors = [
                "#FF4B4B" if kat == kategori_pilih else "lightgray"
                for kat in kategori_summary["KATEGORI"]
            ]
            fig_donut = px.pie(
                kategori_summary,
                names="KATEGORI",
                values="JUMLAH",
                hole=0.45
            )
            fig_donut.update_traces(
                textinfo="percent",
                marker=dict(colors=colors, line=dict(color="white", width=2)),
                textposition="inside",
                showlegend=False
            )
            fig_donut.update_layout(
                height=350,  # donut lebih ringkas
                margin=dict(t=20, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_donut, use_container_width=True, key="fig_donut_kategori")

            # --- tabel ringkas ---
            st.markdown("**Rincian Kategori**")
            tabel_ringkas = kategori_summary[["KATEGORI", "JUMLAH", "PERSENTASE"]].copy()

            # Sort berdasarkan persentase terbesar
            tabel_ringkas = tabel_ringkas.sort_values("PERSENTASE", ascending=False).reset_index(drop=True)

            # Format persen
            tabel_ringkas["PERSENTASE"] = tabel_ringkas["PERSENTASE"].astype(str) + "%"

            # Tambahkan nomor urut
            tabel_ringkas.index = tabel_ringkas.index + 1
            tabel_ringkas.index.name = "No"

            st.table(tabel_ringkas)



        # === Baris baru: Komposisi Jenis Izin per Bulan ===
        st.subheader("Komposisi Jenis Izin per Bulan")
        bulan_for_bar_home = st.selectbox("Pilih Bulan (Halaman Utama)", options=bulan_opts, key="bar_month_home")

        dcomp_home = (dff[dff["BULAN"] == bulan_for_bar_home]
            .groupby("JENIS IZIN", as_index=False)["JUMLAH"].sum()
            .sort_values("JUMLAH", ascending=False)
        )

        fig_home = px.bar(
            dcomp_home.head(15), x="JUMLAH", y="JENIS IZIN",
            orientation="h", text="JUMLAH",
            color="JUMLAH", color_continuous_scale="Reds"
        )
        fig_home.update_traces(texttemplate='%{text}', textposition='outside')
        fig_home.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            xaxis_title="Jumlah", yaxis_title="Jenis Izin"
        )

        st.plotly_chart(fig_home, use_container_width=True, key="fig_komposisi_jenis")

               


    # === Tab 2: Sektor per Bulan ===
    with tab2:
        st.subheader("Perbandingan Sektor per Bulan")
        barmode = st.radio("Mode batang", ["group", "stack"], horizontal=True, key="barmode")
        fig2 = px.bar(
            dff, x="BULAN", y="JUMLAH", color="SEKTOR", barmode=barmode,
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig2.update_layout(
            yaxis_title="Jumlah", xaxis_title="Bulan",
            legend_title="Sektor", hovermode="x unified", plot_bgcolor="white"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # === Tab 3: Tren Top-N Jenis Izin ===
    with tab3:
        st.subheader("Tren Top-N Jenis Izin")
        top_izin = dff.groupby("JENIS IZIN")["JUMLAH"].sum().nlargest(int(top_n)).index
        df_top = dff[dff["JENIS IZIN"].isin(top_izin)]

        fig3 = px.line(
            df_top, x="BULAN", y="JUMLAH", color="JENIS IZIN",
            markers=True, line_shape="spline",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig3.update_traces(line_width=3, mode="lines+markers")
        fig3.update_layout(
            yaxis_title="Jumlah", xaxis_title="Bulan",
            legend_title="Jenis Izin", hovermode="x unified", plot_bgcolor="white"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # === Tab 4: Komposisi Bulan ===
    with tab4:
        st.subheader("Komposisi Jenis Izin per Bulan")
        bulan_for_bar = st.selectbox("Pilih Bulan", options=bulan_opts, key="bar_month")
        dcomp = (dff[dff["BULAN"] == bulan_for_bar]
                .groupby("JENIS IZIN", as_index=False)["JUMLAH"].sum()
                .sort_values("JUMLAH", ascending=False))

        fig4 = px.bar(
            dcomp.head(15), x="JUMLAH", y="JENIS IZIN",
            orientation="h", text="JUMLAH",
            color="JUMLAH", color_continuous_scale="Reds"
        )
        fig4.update_traces(texttemplate='%{text}', textposition='outside')
        fig4.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            xaxis_title="Jumlah", yaxis_title="Jenis Izin"
        )
        st.plotly_chart(fig4, use_container_width=True)

    # === Tab 5: Perbandingan Kategori ===
    with tab5:
        st.subheader("Tren Kategori per Bulan")
        g5 = df[
            (df["SEKTOR"].isin(sel_sektor)) &
            (df["JENIS IZIN"].isin(sel_jenis)) &
            (df["BULAN"].isin(sel_bulan))
        ].groupby(["BULAN", "KATEGORI"], as_index=False)["JUMLAH"].sum()

        fig5 = px.line(g5, x="BULAN", y="JUMLAH", color="KATEGORI", markers=True, line_shape="spline")
        fig5.update_traces(mode="lines+markers", line_width=3)
        fig5.update_layout(
            yaxis_title="Jumlah", xaxis_title="Bulan",
            legend_title="Kategori", hovermode="x unified", plot_bgcolor="white"
        )
        st.plotly_chart(fig5, use_container_width=True)

    # === Tab 6: Tabel & Unduh ===
    with tab6:
        st.subheader("Data Terfilter")
        st.dataframe(dff, use_container_width=True, hide_index=True)
        csv = dff.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Unduh CSV (filter saat ini)", data=csv,
                        file_name="data_filter.csv", mime="text/csv")

        st.divider()






    
   

else:
    st.info("Silakan upload file Excel mentah untuk memulai.")
