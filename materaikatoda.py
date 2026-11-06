""
Screening Material Katoda Na-ion (oksida logam transisi)
Sumber data : Materials Project (insertion_electrodes, working ion = Na)
Output      : tabel kandidat (CSV), tabel funnel (CSV), 2 gambar (PNG)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from pymatgen.core import Composition

# ==========================================
# 1. KONFIGURASI
# ==========================================
API_KEY = os.environ.get("MP_API_KEY", "960HzNNVoTEWuIeNhH9k3uXVwjrrHDo9")

TM_ALL = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Mo", "Nb", "Zr"}

# Kriteria screening (cantumkan dan justifikasi tiap nilai di bagian Metode paper)
V_MIN, V_MAX = 2.0, 4.0     # tegangan rata-rata (V vs Na/Na+)
CAP_MIN      = 100          # kapasitas gravimetrik minimum (mAh/g)
VOL_MAX      = 15           # perubahan volume maksimum (%)
EHULL_MAX    = 0.05         # E_above_hull fase charged/discharged (eV/atom)
MAX_V_STEP   = None         # mis. 1.0 untuk membatasi lompatan tegangan antar-fase; None = nonaktif

TOP_N   = 15
OUT_DIR = "hasil_screening"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# 2. AMBIL DATA DARI MATERIALS PROJECT
# ==========================================
print("--- Screening Material Katoda Na-ion ---")
with MPRester(API_KEY) as mpr:
    docs = mpr.materials.insertion_electrodes.search(working_ion="Na")
print(f"Total elektroda Na terambil: {len(docs)}")


def g(d, name, default=None):
    return getattr(d, name, default)


rows = []
for d in docs:
    rows.append({
        "Material IDs": ", ".join(map(str, g(d, "material_ids", []) or [])),
        "Formula discharged": g(d, "formula_discharge"),
        "Formula charged": g(d, "formula_charge"),
        "Framework": g(d, "framework_formula"),
        "Avg Voltage (V)": g(d, "average_voltage"),
        "Max V Step (V)": g(d, "max_voltage_step"),
        "Capacity (mAh/g)": g(d, "capacity_grav"),
        "Capacity (mAh/cm3)": g(d, "capacity_vol"),
        "Energy Density (Wh/kg)": g(d, "energy_grav"),
        "Energy Density (Wh/L)": g(d, "energy_vol"),
        "Vol Change (%)": (g(d, "max_delta_volume") or 0) * 100,
        "E_hull charge (eV)": g(d, "stability_charge"),
        "E_hull discharge (eV)": g(d, "stability_discharge"),
        "Steps": g(d, "num_steps"),
    })

df_all = pd.DataFrame(rows)
num_cols = [c for c in df_all.columns
            if c not in ("Material IDs", "Formula discharged", "Formula charged", "Framework")]
df_all[num_cols] = df_all[num_cols].apply(pd.to_numeric, errors="coerce")

# Cek ketersediaan data stabilitas
for c in ("E_hull charge (eV)", "E_hull discharge (eV)"):
    print(f"  {c}: terisi {df_all[c].notna().sum()}/{len(df_all)}")


# ==========================================
# 3. FILTER BERTAHAP (FUNNEL)
# ==========================================
def tm_in_framework(formula):
    try:
        els = {str(e) for e in Composition(formula).elements}
        return sorted(els & TM_ALL) if "O" in els else []
    except Exception:
        return []


df_all["Logam Transisi"] = df_all["Framework"].apply(tm_in_framework).apply(lambda x: "-".join(x))

funnel = [("Semua elektroda Na", len(df_all))]
df = df_all.copy()

steps = [
    ("Oksida logam transisi", lambda x: x["Logam Transisi"] != ""),
    (f"Tegangan {V_MIN}-{V_MAX} V", lambda x: x["Avg Voltage (V)"].between(V_MIN, V_MAX)),
    (f"Kapasitas >= {CAP_MIN} mAh/g", lambda x: x["Capacity (mAh/g)"] >= CAP_MIN),
    (f"Perubahan volume <= {VOL_MAX}%", lambda x: x["Vol Change (%)"] <= VOL_MAX),
]
if df_all["E_hull charge (eV)"].notna().any():
    steps.append((f"E_hull charged <= {EHULL_MAX} eV",
                  lambda x: x["E_hull charge (eV)"].fillna(np.inf) <= EHULL_MAX))
if df_all["E_hull discharge (eV)"].notna().any():
    steps.append((f"E_hull discharged <= {EHULL_MAX} eV",
                  lambda x: x["E_hull discharge (eV)"].fillna(np.inf) <= EHULL_MAX))
if MAX_V_STEP is not None:
    steps.append((f"Max voltage step <= {MAX_V_STEP} V",
                  lambda x: x["Max V Step (V)"].fillna(np.inf) <= MAX_V_STEP))

for label, cond in steps:
    df = df[cond(df)]
    funnel.append((label, len(df)))

df_funnel = pd.DataFrame(funnel, columns=["Tahap filter", "Jumlah elektroda"])
print("\n--- FUNNEL FILTER ---")
print(df_funnel.to_string(index=False))
df_funnel.to_csv(f"{OUT_DIR}/funnel_filter.csv", index=False)

# ==========================================
# 4. PERINGKAT KANDIDAT
# ==========================================
if df.empty:
    print("\nTidak ada kandidat lolos. Lihat funnel di atas, tahap mana yang paling ketat,")
    print("lalu longgarkan (mis. VOL_MAX=25, CAP_MIN=80, EHULL_MAX=0.1).")
else:
    # Hapus duplikat (formula discharged sama), simpan yang densitas energinya tertinggi
    df = (df.sort_values("Energy Density (Wh/kg)", ascending=False)
            .drop_duplicates(subset=["Formula discharged", "Formula charged"])
            .reset_index(drop=True))

    # Skor komposit sederhana (min-max), hanya sebagai alat bantu pemeringkatan
    def mm(s):
        return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0 + 0.5

    df["Skor"] = (0.4 * mm(df["Energy Density (Wh/kg)"])
                  + 0.3 * mm(df["Capacity (mAh/g)"])
                  + 0.3 * (1 - mm(df["Vol Change (%)"])))
    df = df.sort_values("Skor", ascending=False).round(3)

    show = ["Material IDs", "Formula discharged", "Logam Transisi", "Avg Voltage (V)",
            "Capacity (mAh/g)", "Energy Density (Wh/kg)", "Vol Change (%)", "Skor"]
    print(f"\n--- TOP {TOP_N} KANDIDAT KATODA ---")
    print(df[show].head(TOP_N).to_string(index=False))

    df.to_csv(f"{OUT_DIR}/kandidat_katoda_na.csv", index=False)
    print(f"\n{len(df)} kandidat unik disimpan di '{OUT_DIR}/kandidat_katoda_na.csv'")

    # ==========================================
    # 5. VISUALISASI
    # ==========================================
    # (a) Tegangan vs kapasitas, warna = perubahan volume
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(df["Capacity (mAh/g)"], df["Avg Voltage (V)"],
                    c=df["Vol Change (%)"], cmap="viridis_r", s=40, edgecolor="k", linewidth=0.3)
    plt.colorbar(sc, label="Perubahan volume (%)")
    for _, r in df.head(5).iterrows():
        ax.annotate(r["Formula discharged"], (r["Capacity (mAh/g)"], r["Avg Voltage (V)"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Kapasitas gravimetrik (mAh/g)")
    ax.set_ylabel("Tegangan rata-rata (V vs Na/Na$^+$)")
    ax.set_title("Kandidat katoda Na-ion hasil screening")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/voltage_vs_capacity.png", dpi=300)
    plt.show()

    # (b) Bar chart top kandidat berdasarkan densitas energi
    top = df.head(TOP_N).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["Formula discharged"], top["Energy Density (Wh/kg)"], color="steelblue")
    ax.set_xlabel("Densitas energi (Wh/kg)")
    ax.set_title(f"Top {len(top)} kandidat berdasarkan skor")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/top_kandidat_energi.png", dpi=300)
    plt.show()

    # (c) Ringkasan per logam transisi (berguna untuk pembahasan paper)
    ringkasan = (df.groupby("Logam Transisi")
                   .agg(Jumlah=("Formula discharged", "count"),
                        Tegangan_rata2=("Avg Voltage (V)", "mean"),
                        Kapasitas_rata2=("Capacity (mAh/g)", "mean"),
                        Energi_maks=("Energy Density (Wh/kg)", "max"))
                   .round(2).sort_values("Jumlah", ascending=False))
    print("\n--- RINGKASAN PER LOGAM TRANSISI ---")
    print(ringkasan.to_string())
    ringkasan.to_csv(f"{OUT_DIR}/ringkasan_per_logam.csv")
