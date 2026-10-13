# -*- coding: utf-8 -*-
"""
GDC Immobiliare - grafici per il report finale. Legge i risultati GIA' prodotti dagli altri script (nessun ricalcolo dei modelli).

Input  (cartella analisi_modelli): out_descr, out_vendite, out_affitti, out_aggregato, out_rendimento  (file .csv)
Output: analisi_modelli/out_report/*.png

Grafici:
  g01_medie_per_citta            canone medio, prezzo di vendita medio e spese condominiali medie per citta'
  g02_rendimento_per_citta       rendimento lordo (canone + spese) e netto (solo canone) per citta'
  g03_heatmap_canone             canone medio per citta' x tipologia
  g04_heatmap_rendimento_netto   rendimento netto per citta' x tipologia (* = pochi annunci)
  g05_effetti_affitti            effetti % di citta' e tipologia sul canone (modello A1) con IC 95%
  g06_effetti_vendite            effetti % di citta' e tipologia sul prezzo di vendita (modello V1) con IC 95%
  g07_effetti_modello_aggregato  effetti % del modello M2 (dati aggregati) con IC 95%
  g08_confronto_tipologie        sconto/premio delle tipologie rispetto al bilocale: affitti contro vendite
  g09_validazione_incrociata     R2 in validazione incrociata dei modelli (affitti, vendite, aggregato)
  g10_importanza_variabili       importanza delle variabili (Random Forest)
Colore blu = parametro significativo (p < 0,05); grigio = non significativo.

Uso:   python grafici_report.py
Dipendenze: pandas, numpy, matplotlib. Nessun accesso a internet.
"""
import argparse, io, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
BLU, ARANCIO, GRIGIO, FONDO, TESTO = "#2a78d6", "#e07b2e", "#9aa0a8", "#fcfcfb", "#1f2328"
ORDINE_TIP = ["Monolocale", "Bilocale", "Trilocale", "Quadrilocale+", "Altro (attico/casa)"]
CREATI, SALTATI = [], []


def leggi(path):
    if not os.path.exists(path):
        return None
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            t = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    p = t.splitlines()[0]
    return pd.read_csv(io.StringIO(t), sep=";" if p.count(";") >= p.count(",") else ",")


def stile(ax, griglia="x"):
    ax.set_facecolor(FONDO)
    ax.grid(axis=griglia, color="#e6e6e3", linewidth=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def salva(fig, out, nome):
    fig.patch.set_facecolor(FONDO); fig.tight_layout()
    path = os.path.join(out, nome + ".png")
    try:
        fig.savefig(path, dpi=140, facecolor=FONDO)
        CREATI.append(nome + ".png")
    except OSError as e:
        SALTATI.append("%s (%s) - file probabilmente aperto in un visualizzatore" % (nome, e))
    plt.close(fig)


def manca(nome, *file):
    SALTATI.append("%s (file mancante: %s)" % (nome, ", ".join(file)))


# ------------------------------------------------------------------ grafici
def g01(base, out):
    a = leggi(os.path.join(base, "out_descr", "descr_affitti_per_citta.csv"))
    v = leggi(os.path.join(base, "out_descr", "descr_vendite_per_citta.csv"))
    s = leggi(os.path.join(base, "out_descr", "descr_affitti_spese_condominiali_per_citta.csv"))
    if a is None or v is None or s is None:
        return manca("g01", "descr_affitti_per_citta.csv", "descr_vendite_per_citta.csv", "descr_affitti_spese_condominiali_per_citta.csv")
    pann = [(a, "Canone medio (EUR/mese)", BLU), (v, "Prezzo di vendita medio (EUR)", ARANCIO), (s, "Spese condominiali medie (EUR/mese)", GRIGIO)]
    fig, axs = plt.subplots(1, 3, figsize=(16, 6.5))
    for ax, (d, tit, col) in zip(axs, pann):
        d = d.dropna(subset=["media"]).sort_values("media")
        ax.barh(d["citta"], d["media"], color=col, alpha=0.85)
        for y, (m, n) in enumerate(zip(d["media"], d["n"])):
            ax.text(m, y, " %s (n=%d)" % (format(int(round(m)), ","), n), va="center", fontsize=7)
        ax.set_title(tit, loc="left", fontsize=10); ax.set_xlim(0, d["media"].max() * 1.3); stile(ax)
        ax.tick_params(axis="y", labelsize=8)
    salva(fig, out, "g01_medie_per_citta")


def g02(base, out):
    k = leggi(os.path.join(base, "out_rendimento", "rendimento_citta.csv"))
    if k is None:
        return manca("g02", "rendimento_citta.csv")
    k = k.sort_values("rendimento_netto_%")
    y = np.arange(len(k))
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(k) + 1.5)))
    for i, (_, r) in enumerate(k.iterrows()):
        bassa = r["affidabilita"] == "bassa"
        if pd.notna(r["rendimento_lordo_%"]):
            ax.barh(i + 0.2, r["rendimento_lordo_%"], height=0.38, color=BLU, alpha=0.4 if bassa else 0.9, hatch="//" if bassa else None, edgecolor="white")
            ax.text(max(r["rendimento_lordo_%"], 0), i + 0.2, " %.1f%%" % r["rendimento_lordo_%"], va="center", fontsize=7)
        ax.barh(i - 0.2, r["rendimento_netto_%"], height=0.38, color=ARANCIO, alpha=0.4 if bassa else 0.9, hatch="//" if bassa else None, edgecolor="white")
        ax.text(max(r["rendimento_netto_%"], 0), i - 0.2, " %.1f%%" % r["rendimento_netto_%"], va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels(["%s (v=%d, a=%d)" % (c, nv, na) for c, nv, na in zip(k["citta"], k["n_vendite"], k["n_affitti"])], fontsize=8)
    ax.set_xlabel("rendimento annuo (%)"); ax.set_title("Rendimento per citta': lordo = canone + spese (blu), netto = solo canone (arancione)", loc="left", fontsize=10)
    ax.text(0.99, 0.02, "tratteggio = pochi annunci (<3 vendite o <3 affitti)\nv = annunci in vendita, a = annunci in affitto", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=TESTO)
    stile(ax); salva(fig, out, "g02_rendimento_per_citta")


def heatmap(base, out, colonna, titolo, nome, fmt, marca_bassa):
    c = leggi(os.path.join(base, "out_rendimento", "rendimento_celle_citta_tipologia.csv"))
    k = leggi(os.path.join(base, "out_rendimento", "rendimento_citta.csv"))
    if c is None or k is None:
        return manca(nome, "rendimento_celle_citta_tipologia.csv", "rendimento_citta.csv")
    citta = list(k.sort_values("rendimento_netto_%", ascending=False)["citta"])
    citta += [x for x in sorted(set(c["citta"])) if x not in citta]
    tip = [t for t in ORDINE_TIP if t in set(c["tipologia"])]
    M = np.full((len(citta), len(tip)), np.nan); B = np.zeros(M.shape, bool)
    for _, r in c.iterrows():
        i, j = citta.index(r["citta"]), tip.index(r["tipologia"])
        M[i, j] = r[colonna]; B[i, j] = r["affidabilita"] == "bassa"
    fig, ax = plt.subplots(figsize=(1.7 * len(tip) + 3, 0.42 * len(citta) + 2))
    cm = plt.get_cmap("Blues").copy(); cm.set_bad("#eceae6")
    vmax = np.nanmax(M) if np.isfinite(M).any() else 1
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cm, aspect="auto", vmin=0, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                chiaro = M[i, j] < 0.55 * vmax
                ax.text(j, i, fmt(M[i, j]) + ("*" if (marca_bassa and B[i, j]) else ""), ha="center", va="center", fontsize=8,
                        color=TESTO if chiaro else "white")
    ax.set_xticks(range(len(tip))); ax.set_xticklabels(tip, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(citta))); ax.set_yticklabels(citta, fontsize=8)
    ax.set_title(titolo + ("   (* = cella con pochi annunci; grigio = nessun dato)" if marca_bassa else "   (grigio = nessun dato)"), loc="left", fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    salva(fig, out, nome)


def etichetta(v, rif=None):
    if v.startswith("log_prezzo_vendita"):
        return "Prezzo di vendita (+10%)"
    if v.startswith("spese"):
        return "Spese condominiali (+10 EUR/mese)"
    nome = v.split("=", 1)[1].split("(vs")[0].strip() if "=" in v else v
    return nome


def forest(base, out, percorso, titolo, nome, colonna_effetto=None):
    d = leggi(os.path.join(base, *percorso))
    if d is None:
        return manca(nome, "/".join(percorso))
    col = colonna_effetto or ("effetto_%_stima" if "effetto_%_stima" in d else "effetto_%")
    lo, hi = col + "_ci95_inf", col + "_ci95_sup"
    if "effetto_%_stima" in d:
        lo, hi = "effetto_%_ci95_inf", "effetto_%_ci95_sup"
    d = d[d["variabile"] != "Intercetta"].dropna(subset=[col]).reset_index(drop=True)
    if d.empty:
        return manca(nome, "/".join(percorso) + " (nessun coefficiente)")
    rifs = sorted(set(x.split("(vs")[1].strip(" )") for x in d["variabile"] if "(vs" in x))
    xmin, xmax = max(-100, min(-60, d[lo].min() - 5)), min(300, max(60, d[hi].max() + 5))
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.36 * len(d) + 1.6)))
    for i, r in d.iterrows():
        y = len(d) - 1 - i
        sig = r["p_value"] < 0.05
        col_pt = BLU if sig else GRIGIO
        a, b = max(r[lo], xmin), min(r[hi], xmax)
        ax.plot([a, b], [y, y], color=col_pt, linewidth=1.6)
        ax.plot([r[col]], [y], "o", color=col_pt, markersize=5)
        if r[hi] > xmax:
            ax.annotate("", xy=(xmax, y), xytext=(xmax - 8, y), arrowprops=dict(arrowstyle="->", color=col_pt))
    ax.axvline(0, color=TESTO, linewidth=0.9)
    ax.set_yticks(range(len(d))); ax.set_yticklabels([etichetta(v) for v in d["variabile"]][::-1], fontsize=8)
    ax.set_xlim(xmin, xmax); ax.set_xlabel("variazione % rispetto al riferimento (intervallo di confidenza 95%)")
    ax.set_title(titolo, loc="left", fontsize=10)
    ax.text(0.99, 0.01, "riferimento: " + " | ".join(rifs) + "\nblu = significativo (p<0,05), grigio = non significativo", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7)
    stile(ax); salva(fig, out, nome)


def g08(base, out):
    a = leggi(os.path.join(base, "out_affitti", "A1_coefficienti.csv"))
    v = leggi(os.path.join(base, "out_vendite", "V1_ols_coefficienti.csv"))
    if a is None or v is None:
        return manca("g08", "A1_coefficienti.csv", "V1_ols_coefficienti.csv")
    def prendi(d):
        d = d[d["variabile"].str.startswith("tip")].copy()
        d["t"] = d["variabile"].map(etichetta)
        return d.set_index("t")
    A, V = prendi(a), prendi(v)
    tip = [t for t in ORDINE_TIP if t in A.index or t in V.index]
    x = np.arange(len(tip)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    for off, d, col, lab in ((-w / 2, A, BLU, "Affitti (canone)"), (w / 2, V, ARANCIO, "Vendite (prezzo)")):
        val = [d["effetto_%_stima"].get(t, np.nan) for t in tip]
        err_lo = [val[i] - d["effetto_%_ci95_inf"].get(t, np.nan) for i, t in enumerate(tip)]
        err_hi = [d["effetto_%_ci95_sup"].get(t, np.nan) - val[i] for i, t in enumerate(tip)]
        sig = [d["p_value"].get(t, 1) < 0.05 for t in tip]
        ax.bar(x + off, val, w, color=col, alpha=0.9, label=lab, edgecolor="white")
        ax.errorbar(x + off, val, yerr=[err_lo, err_hi], fmt="none", ecolor=TESTO, elinewidth=0.9, capsize=3)
        for xi, vi, s in zip(x + off, val, sig):
            if np.isfinite(vi):
                ax.text(xi, vi, ("*" if s else ""), ha="center", va="bottom" if vi >= 0 else "top", fontsize=12)
    ax.axhline(0, color=TESTO, linewidth=0.9); ax.set_xticks(x); ax.set_xticklabels(tip, fontsize=9)
    ax.set_ylabel("variazione % rispetto al bilocale"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Tipologia: sconto o premio rispetto al bilocale (barre = IC 95%, * = p<0,05)", loc="left", fontsize=10)
    stile(ax, "y"); salva(fig, out, "g08_confronto_tipologie")


def g09(base, out):
    fonti = [("Affitti: canone", ["out_affitti", "A5_confronto_modelli_cv_Y.csv"]),
             ("Affitti: canone + spese", ["out_affitti", "A5_confronto_modelli_cv_Y2.csv"]),
             ("Vendite: prezzo", ["out_vendite", "V2_confronto_modelli_cv.csv"]),
             ("Dati aggregati (WLS)", ["out_aggregato", "confronto_modelli_cv.csv"])]
    dati = [(t, leggi(os.path.join(base, *p))) for t, p in fonti]
    dati = [(t, d) for t, d in dati if d is not None]
    if not dati:
        return manca("g09", "file di confronto modelli")
    fig, axs = plt.subplots(1, len(dati), figsize=(4.8 * len(dati), 4.8), squeeze=False)
    for ax, (t, d) in zip(axs[0], dati):
        d = d.copy(); d["modello"] = d["modello"].str.replace(" (nessun predittore)", "", regex=False).str.slice(0, 34)
        cols = [ARANCIO if m.lower().startswith("media") else BLU for m in d["modello"]]
        ax.barh(d["modello"][::-1], d["R2_CV"][::-1], xerr=d["R2_CV_dev_std"][::-1], color=cols[::-1], alpha=0.9, error_kw=dict(elinewidth=0.9, capsize=2))
        ax.axvline(0, color=TESTO, linewidth=0.9); ax.set_title(t, loc="left", fontsize=10); ax.set_xlabel("R2 in validazione incrociata")
        ax.tick_params(axis="y", labelsize=7); stile(ax)
    fig.suptitle("Capacita' predittiva: R2 in validazione incrociata (arancione = media, nessun predittore; valori <= 0 = nessun potere predittivo)", x=0.01, ha="left", fontsize=9)
    salva(fig, out, "g09_validazione_incrociata")


def g10(base, out):
    fonti = [("Affitti: canone", ["out_affitti", "A5_importanza_variabili_Y.csv"]),
             ("Affitti: canone + spese", ["out_affitti", "A5_importanza_variabili_Y2.csv"]),
             ("Vendite: prezzo", ["out_vendite", "V2_importanza_variabili.csv"])]
    dati = [(t, leggi(os.path.join(base, *p))) for t, p in fonti]
    dati = [(t, d) for t, d in dati if d is not None]
    if not dati:
        return manca("g10", "file di importanza delle variabili")
    fig, axs = plt.subplots(1, len(dati), figsize=(4.8 * len(dati), 3.6), squeeze=False)
    for ax, (t, d) in zip(axs[0], dati):
        d = d.sort_values("importanza_calo_R2")
        ax.barh(d["variabile"], d["importanza_calo_R2"], color=BLU, alpha=0.9)
        for y, v in enumerate(d["importanza_calo_R2"]):
            ax.text(v, y, " %.2f" % v, va="center", fontsize=8)
        ax.set_title(t, loc="left", fontsize=10); ax.set_xlabel("importanza (calo di R2 permutando la variabile)"); stile(ax)
    salva(fig, out, "g10_importanza_variabili")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cartella", default="analisi_modelli")
    a = ap.parse_args()
    base = a.cartella if os.path.isabs(a.cartella) else os.path.join(AQUI, a.cartella)
    if not os.path.isdir(base):
        print("Cartella non trovata:", base); return 1
    out = os.path.join(base, "out_report"); os.makedirs(out, exist_ok=True)
    passi = [lambda: g01(base, out), lambda: g02(base, out),
             lambda: heatmap(base, out, "canone_medio", "Canone medio (EUR/mese) per citta' e tipologia", "g03_heatmap_canone", lambda x: "%d" % round(x), True),
             lambda: heatmap(base, out, "rendimento_netto_%", "Rendimento netto (canone x 12 / prezzo, %) per citta' e tipologia", "g04_heatmap_rendimento_netto", lambda x: "%.1f" % x, True),
             lambda: forest(base, out, ["out_affitti", "A1_coefficienti.csv"], "Affitti (modello A1): effetto di citta' e tipologia sul canone", "g05_effetti_affitti"),
             lambda: forest(base, out, ["out_vendite", "V1_ols_coefficienti.csv"], "Vendite (modello V1): effetto di citta' e tipologia sul prezzo di vendita", "g06_effetti_vendite"),
             lambda: forest(base, out, ["out_aggregato", "M2_coefficienti.csv"], "Dati aggregati (modello M2): effetti sul canone medio", "g07_effetti_modello_aggregato"),
             lambda: g08(base, out), lambda: g09(base, out), lambda: g10(base, out)]
    for p in passi:
        try:
            p()
        except Exception as e:                       # un grafico che fallisce non blocca gli altri
            SALTATI.append("%s: %s: %s" % (getattr(p, "__name__", "grafico"), type(e).__name__, e))
    print("Grafici creati (%d) in %s:" % (len(CREATI), out))
    for c in CREATI:
        print("  ", c)
    if SALTATI:
        print("\nNon creati:")
        for s in SALTATI:
            print("  ", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
