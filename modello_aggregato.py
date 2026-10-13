# -*- coding: utf-8 -*-
"""
GDC Immobiliare - regressione PESATA (WLS) sul file aggregato output_aggregato.csv (una riga = citta' x tipologia).

Input : analisi_modelli/output_aggregato.csv   (prodotto da aggrega_output.py)
Output: analisi_modelli/out_aggregato/

Y = log(canone_medio)   (gli effetti si leggono in %)
Pesi = n_affitti (la media di una cella con piu' annunci e' piu' affidabile). Errori standard robusti HC3.

Modelli:
  M0  citta' + tipologia                                          (riferimento, senza informazioni di vendita e spese)
  M1  tipologia + log(prezzo_vendita_medio) + spese_condominiali_medie
  M2  citta' + tipologia + log(prezzo_vendita_medio) + spese_condominiali_medie
Tutti i modelli sono stimati sullo STESSO campione (righe con n_affitti >= soglia, e con prezzo di vendita e spese disponibili),
cosi' i confronti sono omogenei. Le righe escluse sono in righe_escluse.csv.
Confronto con validazione incrociata (5 fold x 3 ripetizioni) contro la sola media pesata.

Interpretazione dei coefficienti (colonna effetto_%):
  variabili categoriche : variazione % del canone rispetto al livello di riferimento
  log(prezzo_vendita)   : variazione % del canone per un +10% del prezzo di vendita medio (il coefficiente e' l'elasticita')
  spese (decine di EUR) : variazione % del canone per +10 EUR/mese di spese condominiali

Uso:   python modello_aggregato.py
       python modello_aggregato.py --min-n 3        (n minimo di annunci d'affitto per cella; default 3)
       python modello_aggregato.py --nessun-peso    (OLS non pesata)
Dipendenze: pandas, numpy, scipy, matplotlib (incluse in Anaconda). Nessun accesso a internet.
"""
import argparse, io, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
AQUI = os.path.dirname(os.path.abspath(__file__))
BLU, GRIGIO, FONDO = "#2a78d6", "#8a8f98", "#fcfcfb"


# ------------------------------------------------------------------ utilita'
def leggi(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            t = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    prima = t.splitlines()[0]
    sep = ";" if prima.count(";") >= prima.count(",") else ","
    return pd.read_csv(io.StringIO(t), sep=sep)


def scrivi_csv(df, path):
    df.to_csv(path, sep=";", index=False, encoding="utf-8-sig")


def costruisci_X(d, cat, num, rif):
    """Matrice di progetto: intercetta + dummy (livello di riferimento escluso) + variabili numeriche."""
    cols, nomi, tipo = [np.ones(len(d))], ["Intercetta"], ["int"]
    for c in cat:
        for lv in sorted(set(d[c])):
            if lv == rif[c]:
                continue
            cols.append((d[c] == lv).to_numpy(float)); nomi.append("%s = %s (vs %s)" % (c, lv, rif[c])); tipo.append("cat")
    for c in num:
        cols.append(d[c].to_numpy(float)); nomi.append(c); tipo.append("num")
    return np.column_stack(cols), nomi, tipo


def wls(X, y, w):
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    XtXi = np.linalg.pinv(Xw.T @ Xw)
    b = XtXi @ Xw.T @ yw
    e = yw - Xw @ b
    h = np.clip(np.einsum("ij,jk,ik->i", Xw, XtXi, Xw), 0, 0.999)
    meat = (Xw * (e / (1 - h))[:, None]).T @ (Xw * (e / (1 - h))[:, None])
    V = XtXi @ meat @ XtXi
    n, rk = len(y), np.linalg.matrix_rank(Xw)
    ybar = np.sum(w * y) / np.sum(w)
    rss, tss = float(np.sum(e ** 2)), float(np.sum(w * (y - ybar) ** 2))
    return {"b": b, "se": np.sqrt(np.clip(np.diag(V), 0, None)), "rss": rss, "tss": tss, "n": n, "rk": rk,
            "df": n - rk, "r2": 1 - rss / tss if tss > 0 else np.nan}


def trasforma(nome, tipo, b, lo, hi):
    """Coefficiente (e IC) -> effetto % interpretabile."""
    if tipo == "cat":
        f = lambda v: (np.exp(v) - 1) * 100
    elif nome.startswith("log_prezzo_vendita"):
        f = lambda v: (1.1 ** v - 1) * 100
    elif nome.startswith("spese"):
        f = lambda v: (np.exp(v) - 1) * 100
    else:
        return np.nan, np.nan, np.nan
    return f(b), f(lo), f(hi)


def tabella(m, nomi, tipo):
    righe = []
    for i, nome in enumerate(nomi):
        b, se = m["b"][i], m["se"][i]
        if m["df"] > 0 and se > 0:
            t = b / se; p = 2 * stats.t.sf(abs(t), m["df"]); q = stats.t.ppf(0.975, m["df"]); lo, hi = b - q * se, b + q * se
        else:
            t = p = lo = hi = np.nan
        e, elo, ehi = trasforma(nome, tipo[i], b, lo, hi) if tipo[i] != "int" else (np.nan,) * 3
        righe.append({"variabile": nome, "coef": b, "errore_std_HC3": se, "t": t, "p_value": p, "coef_ci95_inf": lo, "coef_ci95_sup": hi,
                      "effetto_%": e, "effetto_%_ci95_inf": elo, "effetto_%_ci95_sup": ehi})
    return pd.DataFrame(righe)


def test_fattore(d, y, w, cat, num, rif, fattore):
    """F parziale (non robusto) per un fattore categorico: modello completo vs senza il fattore."""
    Xf, _, _ = costruisci_X(d, cat, num, rif)
    Xr, _, _ = costruisci_X(d, [c for c in cat if c != fattore], num, rif)
    f, r = wls(Xf, y, w), wls(Xr, y, w)
    q = f["rk"] - r["rk"]
    if q <= 0 or f["df"] <= 0:
        return {"fattore": fattore, "gradi_liberta": q, "F": np.nan, "p_value": np.nan}
    F = ((r["rss"] - f["rss"]) / q) / (f["rss"] / f["df"])
    return {"fattore": fattore, "gradi_liberta": q, "F": F, "p_value": stats.f.sf(F, q, f["df"])}


def r2_pesato(y, p, w):
    ybar = np.sum(w * y) / np.sum(w)
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - ybar) ** 2)


def cv(d, y, w, cat, num, rif, reps=3, k=5):
    X, _, _ = costruisci_X(d, cat, num, rif) if (cat or num) else (np.ones((len(d), 1)), 0, 0)
    out = []
    for rep in range(reps):
        p = np.zeros(len(y))
        for tr, te in KFold(k, shuffle=True, random_state=rep).split(X):
            m = wls(X[tr], y[tr], w[tr])
            p[te] = X[te] @ m["b"]
        out.append((r2_pesato(y, p, w), float(np.average(np.abs(np.exp(y) - np.exp(p)), weights=w))))
    a = np.array(out)
    return a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cartella", default="analisi_modelli")
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--nessun-peso", action="store_true")
    a = ap.parse_args()
    base = a.cartella if os.path.isabs(a.cartella) else os.path.join(AQUI, a.cartella)
    f_in = os.path.join(base, "output_aggregato.csv")
    if not os.path.exists(f_in):
        print("File non trovato:", f_in, "(esegui prima aggrega_output.py)"); return 1
    out = os.path.join(base, "out_aggregato"); os.makedirs(out, exist_ok=True)

    D = leggi(f_in)
    for c in ("n_vendite", "n_affitti", "n_spese", "prezzo_vendita_medio", "canone_medio", "spese_condominiali_medie"):
        D[c] = pd.to_numeric(D[c], errors="coerce")
    D["motivo_esclusione"] = ""
    D.loc[D["canone_medio"].isna() | (D["n_affitti"].fillna(0) < 1), "motivo_esclusione"] = "nessun affitto"
    D.loc[(D["motivo_esclusione"] == "") & (D["n_affitti"] < a.min_n), "motivo_esclusione"] = "meno di %d annunci d'affitto" % a.min_n
    D.loc[(D["motivo_esclusione"] == "") & (D["prezzo_vendita_medio"].isna() | (D["n_vendite"].fillna(0) < 1)), "motivo_esclusione"] = "prezzo di vendita mancante"
    D.loc[(D["motivo_esclusione"] == "") & (D["spese_condominiali_medie"].isna() | (D["n_spese"].fillna(0) < 1)), "motivo_esclusione"] = "spese condominiali mancanti"
    scrivi_csv(D[D.motivo_esclusione != ""], os.path.join(out, "righe_escluse.csv"))
    d = D[D.motivo_esclusione == ""].copy().reset_index(drop=True)
    R = ["REGRESSIONE %s SU output_aggregato.csv" % ("NON PESATA" if a.nessun_peso else "PESATA (pesi = n_affitti)"),
         "Y = log(canone_medio) | righe nel file: %d | usate: %d | escluse: %d (vedi righe_escluse.csv)" % (len(D), len(d), len(D) - len(d)),
         "Motivi di esclusione: " + "; ".join("%s: %d" % (k, v) for k, v in D[D.motivo_esclusione != ""].motivo_esclusione.value_counts().items())]
    if len(d) < 10:
        R.append("Troppe poche righe per stimare i modelli."); open(os.path.join(out, "00_riepilogo_modello_aggregato.txt"), "w", encoding="utf-8").write("\n".join(R))
        print("\n".join(R)); return 1

    d["log_prezzo_vendita"] = np.log(d["prezzo_vendita_medio"])
    d["spese_10eur"] = d["spese_condominiali_medie"] / 10.0
    y = np.log(d["canone_medio"].to_numpy(float))
    w = np.ones(len(d)) if a.nessun_peso else d["n_affitti"].to_numpy(float)
    rif = {"tipologia": "Bilocale" if (d.tipologia == "Bilocale").any() else d.tipologia.value_counts().index[0],
           "citta": d.groupby("citta")["n_affitti"].sum().idxmax()}
    R += ["Livelli di riferimento: tipologia = %s | citta' = %s (la piu' numerosa)" % (rif["tipologia"], rif["citta"]),
          "Citta' nel campione: %d | tipologie: %d" % (d.citta.nunique(), d.tipologia.nunique()), ""]

    NUM = ["log_prezzo_vendita", "spese_10eur"]
    modelli = [("M0", "M0. citta' + tipologia", ["citta", "tipologia"], []),
               ("M1", "M1. tipologia + log(prezzo vendita medio) + spese condominiali", ["tipologia"], NUM),
               ("M2", "M2. citta' + tipologia + log(prezzo vendita medio) + spese condominiali", ["citta", "tipologia"], NUM)]
    ris, sintesi = {}, []
    for sigla, titolo, cat, num in modelli:
        X, nomi, tipo = costruisci_X(d, cat, num, rif)
        m = wls(X, y, w)
        R += ["=" * 78, titolo, "=" * 78]
        if m["df"] <= 0:
            R += ["Modello non stimabile: %d parametri per %d righe." % (m["rk"], m["n"]), ""]
            sintesi.append({"modello": sigla, "n_righe": m["n"], "parametri": m["rk"], "R2_pesato": np.nan, "R2_corretto": np.nan}); continue
        r2c = 1 - (1 - m["r2"]) * (m["n"] - 1) / m["df"]
        T = tabella(m, nomi, tipo)
        ris[sigla] = (m, X, nomi, tipo)
        scrivi_csv(T.round(5), os.path.join(out, "%s_coefficienti.csv" % sigla))
        R += ["righe %d | parametri %d | R2 pesato %.3f | R2 corretto %.3f" % (m["n"], m["rk"], m["r2"], r2c), "",
              "Coefficienti (errori robusti HC3); effetto_% = variazione % del canone (vedi intestazione dello script):",
              T[["variabile", "coef", "p_value", "effetto_%", "effetto_%_ci95_inf", "effetto_%_ci95_sup"]].round(3).to_string(index=False), ""]
        tf = pd.DataFrame([test_fattore(d, y, w, cat, num, rif, c) for c in cat if c in ("citta", "tipologia")])
        if len(tf):
            R += ["Significativita' dei fattori categorici (test F parziale, non robusto):", tf.round(4).to_string(index=False), ""]
        sintesi.append({"modello": sigla, "n_righe": m["n"], "parametri": m["rk"], "R2_pesato": round(m["r2"], 4), "R2_corretto": round(r2c, 4)})

    # ---- validazione incrociata
    cvr = []
    r, s, mae = cv(d, y, w, [], [], rif)
    cvr.append({"modello": "Media pesata (nessun predittore)", "R2_CV": r, "R2_CV_dev_std": s, "MAE_EUR": mae})
    for sigla, titolo, cat, num in modelli:
        if sigla in ris:
            r, s, mae = cv(d, y, w, cat, num, rif)
            cvr.append({"modello": titolo.split(". ")[0] + " " + titolo.split(". ")[1][:60], "R2_CV": r, "R2_CV_dev_std": s, "MAE_EUR": mae})
    C = pd.DataFrame(cvr).round(3)
    scrivi_csv(C, os.path.join(out, "confronto_modelli_cv.csv"))
    S = pd.DataFrame(sintesi)
    scrivi_csv(S, os.path.join(out, "confronto_modelli_in_campione.csv"))
    R += ["=" * 78, "CONFRONTO MODELLI", "=" * 78, "In campione:", S.to_string(index=False), "",
          "Validazione incrociata 5 fold x 3 (R2 pesato su scala log; MAE in euro, pesato):", C.to_string(index=False), "",
          "Lettura: un R2 in campione molto piu' alto di R2_CV indica sovradattamento (pochi dati rispetto ai parametri).",
          "Righe usate in righe_usate.csv."]
    scrivi_csv(d.drop(columns=["motivo_esclusione"]), os.path.join(out, "righe_usate.csv"))

    # ---- grafico osservato vs previsto (modelli stimati)
    mostra = [s for s in ("M1", "M2") if s in ris]
    if mostra:
        fig, axs = plt.subplots(1, len(mostra), figsize=(5.6 * len(mostra), 5), squeeze=False)
        for ax, sigla in zip(axs[0], mostra):
            m, X, _, _ = ris[sigla]
            prev, oss = np.exp(X @ m["b"]), np.exp(y)
            ax.scatter(prev, oss, s=14 + 6 * np.sqrt(w), color=BLU, alpha=0.6, edgecolor="none")
            lim = [min(prev.min(), oss.min()) * 0.9, max(prev.max(), oss.max()) * 1.05]
            ax.plot(lim, lim, color=GRIGIO, linewidth=1)
            ax.set_xlabel("canone medio previsto (EUR/mese)"); ax.set_ylabel("canone medio osservato (EUR/mese)")
            ax.set_title("%s (R2 pesato %.2f)" % (sigla, m["r2"]), loc="left", fontsize=11)
            ax.set_facecolor(FONDO); ax.grid(color="#e6e6e3", linewidth=0.7); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.patch.set_facecolor(FONDO); fig.tight_layout()
        fig.savefig(os.path.join(out, "osservato_vs_previsto.png"), dpi=140, facecolor=FONDO); plt.close(fig)

    open(os.path.join(out, "00_riepilogo_modello_aggregato.txt"), "w", encoding="utf-8").write("\n".join(R))
    print("\n".join(R))
    print("\nSalvato in:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
