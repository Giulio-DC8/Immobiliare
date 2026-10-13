# -*- coding: utf-8 -*-
"""
GDC Immobiliare - analisi dei PREZZI (in euro) su vendite e affitti, due file separati.

Input  (cartella analisi_modelli): output_vendite.csv, output_affitti.csv
Output (cartella analisi_modelli):
  out_descr/    statistiche descrittive, test di differenza tra gruppi, boxplot (solo prezzi in euro)
  out_vendite/  modelli sulle vendite   Y = prezzo di vendita
  out_affitti/  modelli sugli affitti   Y = canone mensile ; Y2 = canone + spese condominiali

Modelli (X = citta' + tipologia, nessun mq):
  OLS su log(Y) con errori robusti HC3, Random Forest e Gradient Boosting come confronto (validazione incrociata 5 fold x3).
I file di input sono considerati GIA' PULITI: nessuna rimozione di duplicati, di annunci rif=NO o di valori estremi. Si scartano
solo le righe vuote e quelle senza prezzo valido (non analizzabili).
Scelte: citta' NON accorpate (X = citta' + tipologia); Y2 calcolato solo per gli annunci con
spese condominiali indicate.

Uso:   python analisi_prezzi.py
Dipendenze: pandas, numpy, scipy, scikit-learn, matplotlib (incluse in Anaconda). Nessun accesso a internet.
"""
import argparse, io, os, re, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
AQUI = os.path.dirname(os.path.abspath(__file__))
CORREZIONI = {"verese": "Varese", "mestra": "Mestre", "pesarourbino": "Pesaro", "pesaro urbino": "Pesaro",
              "san raffaele": "Milano"}
BLU, GRIGIO, FONDO = "#2a78d6", "#8a8f98", "#fcfcfb"


# ------------------------------------------------------------------ utilita'
def leggi_csv(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            txt = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    prima = txt.split("\n", 1)[0]
    sep = ";" if prima.count(";") > prima.count(",") else ","
    return pd.read_csv(io.StringIO(txt), sep=sep, dtype=str, keep_default_na=False), sep


def numerico(s, sep):
    s = s.astype(str).str.strip()
    if sep == ";":
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def citta_da_file(f, fallback=""):
    f = str(f or "").strip()
    if not f:
        return fallback
    base = re.sub(r"\.[A-Za-z0-9]+$", "", f)
    base = re.sub(r"_(vendita|affitto).*$", "", base, flags=re.I)
    base = re.sub(r"\d+$", "", base).replace("_", " ").strip().lower()
    return CORREZIONI.get(base, base.title()) if base else fallback


def tipologia_acc(tip, locali):
    t = str(tip or "").strip().lower()
    for k, v in (("monolocale", "Monolocale"), ("bilocale", "Bilocale"), ("trilocale", "Trilocale"),
                 ("quadrilocale", "Quadrilocale+"), ("pentalocale", "Quadrilocale+"), ("plurilocale", "Quadrilocale+")):
        if t.startswith(k):
            return v
    if t.startswith(("attico", "casa", "villa")) or "due piani" in t:
        return "Altro (attico/casa)"
    if pd.notna(locali):                       # "Appartamento" generico: uso il numero di locali
        return {1: "Monolocale", 2: "Bilocale", 3: "Trilocale"}.get(int(locali), "Quadrilocale+")
    return "Altro (attico/casa)"


def raggruppa(serie, min_n):
    n = serie.value_counts()
    return serie.where(~serie.isin(set(n[n < min_n].index)), "Altre")


def stile(ax):
    ax.set_facecolor(FONDO)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.7)
    ax.set_axisbelow(True)


def salva_fig(fig, path):
    fig.patch.set_facecolor(FONDO)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=FONDO)
    plt.close(fig)


def scrivi_csv(df, path):
    df.to_csv(path, sep=";", index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------ OLS con errori robusti HC3
class OLS:
    def __init__(self, df, y, cat, ref):
        self.cat, self.ref, self.yname = list(cat), ref, y
        d = df.dropna(subset=[y] + self.cat).copy()
        self.livelli = {}
        cols, self.nomi = [np.ones(len(d))], ["Intercetta"]
        for c in self.cat:
            lv = d[c].value_counts().index.tolist()
            rif = ref.get(c, lv[0])
            altri = [x for x in sorted(set(d[c])) if x != rif]
            self.livelli[c] = (rif, altri)
            for x in altri:
                cols.append((d[c] == x).to_numpy(float)); self.nomi.append("%s = %s (vs %s)" % (c, x, rif))
        X, yv = np.column_stack(cols), d[y].to_numpy(float)
        self.n, self.k = X.shape
        self.beta = np.linalg.lstsq(X, yv, rcond=None)[0]
        self.fitted, self.y = X @ self.beta, yv
        self.resid = yv - self.fitted
        self.rss = float(self.resid @ self.resid)
        self.tss = float(((yv - yv.mean()) ** 2).sum())
        self.df_res = self.n - self.k
        self.r2 = 1 - self.rss / self.tss
        self.adj_r2 = 1 - (1 - self.r2) * (self.n - 1) / self.df_res
        XtXi = np.linalg.pinv(X.T @ X)
        h = np.einsum("ij,jk,ik->i", X, XtXi, X)
        w = (self.resid / (1 - np.clip(h, 0, 0.999))) ** 2
        self.se = np.sqrt(np.diag(XtXi @ (X.T * w) @ X @ XtXi))
        self.sigma2 = self.rss / self.df_res
        self.F = ((self.tss - self.rss) / (self.k - 1)) / self.sigma2
        self.pF = 1 - stats.f.cdf(self.F, self.k - 1, self.df_res)
        self.data = d

    def tabella(self):
        t = self.beta / self.se
        p = 2 * (1 - stats.t.cdf(np.abs(t), self.df_res))
        q = stats.t.ppf(0.975, self.df_res)
        out = pd.DataFrame({"variabile": self.nomi, "coef": self.beta, "errore_std_HC3": self.se, "p_value": p,
                            "ci95_inf": self.beta - q * self.se, "ci95_sup": self.beta + q * self.se})
        for c in ("coef", "ci95_inf", "ci95_sup"):
            out["effetto_%_" + c.replace("coef", "stima")] = (np.exp(out[c]) - 1) * 100
        out.loc[out.variabile == "Intercetta", [c for c in out.columns if c.startswith("effetto")]] = np.nan
        return out

    def test_fattori(self):
        righe = []
        for f in self.cat:
            sotto = OLS(self.data, self.yname, [c for c in self.cat if c != f], self.ref) if len(self.cat) > 1 else None
            rss0 = sotto.rss if sotto else self.tss
            k0 = sotto.k if sotto else 1
            dfn = self.k - k0
            F = ((rss0 - self.rss) / dfn) / self.sigma2
            righe.append({"fattore": f, "gradi_liberta": dfn, "F": F, "p_value": 1 - stats.f.cdf(F, dfn, self.df_res)})
        return pd.DataFrame(righe)

    def riepilogo(self, titolo):
        tb = self.tabella()
        r = ["=" * 78, titolo, "=" * 78,
             "osservazioni %d | parametri %d | R2 %.3f | R2 corretto %.3f | F %.2f (p=%.4g)" %
             (self.n, self.k, self.r2, self.adj_r2, self.F, self.pF),
             "livello di riferimento: " + "; ".join("%s=%s" % (c, self.livelli[c][0]) for c in self.cat), "",
             "Effetto % sul prezzo rispetto al livello di riferimento (stima, IC 95%):"]
        t = tb[tb.variabile != "Intercetta"][["variabile", "effetto_%_stima", "effetto_%_ci95_inf", "effetto_%_ci95_sup", "p_value"]]
        r.append(t.round(3).to_string(index=False))
        r += ["", "Significativita' dei fattori (test F parziale):", self.test_fattori().round(4).to_string(index=False)]
        return "\n".join(r)


# ------------------------------------------------------------------ validazione incrociata
def prepro(cat):
    return ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat)])


def cv_modello(pipe, X, ylog, reps=3, k=5):
    r2s, maes = [], []
    for s in range(reps):
        p = cross_val_predict(pipe, X, ylog, cv=KFold(k, shuffle=True, random_state=42 + s))
        r2s.append(r2_score(ylog, p)); maes.append(mean_absolute_error(np.exp(ylog), np.exp(p)))
    return float(np.mean(r2s)), float(np.std(r2s)), float(np.mean(maes))


def confronto_ml(d, ycol, cat, titolo_png, path_csv, path_imp, path_png, etichetta):
    X, y = d[cat].astype(str), np.log(d[ycol].to_numpy(float))
    modelli = {
        "Media (nessun predittore)": make_pipeline(prepro(cat), DummyRegressor()),
        "OLS lineare": make_pipeline(prepro(cat), LinearRegression()),
        "Random Forest": make_pipeline(prepro(cat), RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=42, n_jobs=-1)),
        "Gradient Boosting": make_pipeline(prepro(cat), GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8, random_state=42)),
    }
    righe = []
    for nome, p in modelli.items():
        r2m, r2s, mae = cv_modello(p, X, y)
        righe.append({"Y": etichetta, "modello": nome, "R2_CV": r2m, "R2_CV_dev_std": r2s, "MAE_EUR": mae})
    t = pd.DataFrame(righe).round(3)
    scrivi_csv(t, path_csv)
    imp = pd.Series(0.0, index=cat)
    for tr, te in KFold(5, shuffle=True, random_state=1).split(X):
        rf = make_pipeline(prepro(cat), RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1))
        rf.fit(X.iloc[tr], y[tr])
        imp += pd.Series(permutation_importance(rf, X.iloc[te], y[te], n_repeats=10, random_state=0, scoring="r2").importances_mean, index=cat) / 5
    ti = imp.round(4).reset_index(); ti.columns = ["variabile", "importanza_calo_R2"]
    scrivi_csv(ti, path_imp)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.barh(t["modello"][::-1], t["R2_CV"][::-1], color=BLU); ax1.set_xlabel("R2 in validazione incrociata")
    ax1.set_title("Confronto modelli - " + titolo_png, loc="left", fontsize=10); stile(ax1)
    ax2.barh(ti["variabile"][::-1], ti["importanza_calo_R2"][::-1], color=BLU); ax2.set_xlabel("importanza (calo di R2)")
    ax2.set_title("Variabili piu' importanti", loc="left", fontsize=10); stile(ax2)
    salva_fig(fig, path_png)
    return t, ti


def boxplot_gruppi(d, gr, col, titolo, ylabel, path, min_n=3):
    cnt = d.groupby(gr)[col].size()
    ordine = d.groupby(gr)[col].median().sort_values(ascending=False)
    ordine = ordine[cnt[ordine.index] >= min_n]
    if len(ordine) == 0:
        return
    fig, ax = plt.subplots(figsize=(max(6.5, 0.8 * len(ordine) + 2), 5))
    dati = [d.loc[d[gr] == c, col] for c in ordine.index]
    bp = ax.boxplot(dati, patch_artist=True, medianprops=dict(color="#12326a", linewidth=1.6),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=GRIGIO, markeredgecolor=GRIGIO))
    for b in bp["boxes"]:
        b.set(facecolor=BLU, alpha=0.55, edgecolor=BLU)
    ax.set_xticks(range(1, len(ordine) + 1))
    ax.set_xticklabels(["%s\n(n=%d)" % (c, len(x)) for c, x in zip(ordine.index, dati)], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(titolo, loc="left", fontsize=11); stile(ax)
    salva_fig(fig, path)


def statistiche(d, col, gr):
    g = d.groupby(gr)[col]
    t = g.agg(n="size", media="mean", mediana="median", dev_std="std", p25=lambda s: s.quantile(.25),
              p75=lambda s: s.quantile(.75), minimo="min", massimo="max")
    return t.sort_values("mediana", ascending=False).round(1).reset_index()


def test_gruppi(d, col, gr, etichetta_y, fattore):
    gruppi = [g[col].to_numpy(float) for _, g in d.groupby(gr) if len(g) >= 3]
    if len(gruppi) < 2:
        return None
    kw = stats.kruskal(*gruppi)
    lg = [np.log(x) for x in gruppi]
    an = stats.f_oneway(*lg)
    tutti = np.concatenate(lg)
    ssb = sum(len(x) * (x.mean() - tutti.mean()) ** 2 for x in lg)
    return {"Y": etichetta_y, "fattore": fattore, "gruppi_con_n>=3": len(gruppi), "Kruskal_H": kw.statistic, "Kruskal_p": kw.pvalue,
            "ANOVA_F_su_log": an.statistic, "ANOVA_p": an.pvalue, "eta2": ssb / ((tutti - tutti.mean()) ** 2).sum()}


# ------------------------------------------------------------------ preparazione dati
def prepara(path, operazione, a, rapporto):
    df, sep = leggi_csv(path)
    n0 = len(df)
    df = df[df["id_annuncio"].str.strip() != ""].copy()
    rapporto.append("%s: righe nel file %d | righe vuote scartate %d" % (os.path.basename(path), n0, n0 - len(df)))
    for c in ("prezzo", "mq", "locali", "spese_mese", "spese_anno"):
        if c in df:
            df[c] = numerico(df[c], sep)
    df["citta"] = [citta_da_file(f, c) for f, c in zip(df.get("file_sorgente", ""), df.get("citta_area", ""))]
    df["tip"] = [tipologia_acc(t, l) for t, l in zip(df["tipologia"], df["locali"])]
    ok = df["prezzo"] > 0
    rapporto.append("  prezzo mancante o nullo: %d" % (~ok).sum()); df = df[ok]
    df["citta_m"] = df["citta"]          # citta' NON accorpate
    rapporto.append("  campione finale: %d annunci in %d citta' (nessun accorpamento)" % (len(df), df["citta"].nunique()))
    return df


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cartella", default="analisi_modelli")
    a = ap.parse_args()
    base = a.cartella if os.path.isabs(a.cartella) else os.path.join(AQUI, a.cartella)
    od, ov, oa = (os.path.join(base, x) for x in ("out_descr", "out_vendite", "out_affitti"))
    for p in (od, ov, oa):
        os.makedirs(p, exist_ok=True)
    fv, fa = os.path.join(base, "output_vendite.csv"), os.path.join(base, "output_affitti.csv")
    for f in (fv, fa):
        if not os.path.exists(f):
            print("File non trovato:", f); return 1

    rap_v, rap_a = [], []
    V = prepara(fv, "vendita", a, rap_v)
    A = prepara(fa, "affitto", a, rap_a)
    rapporto = ["PREPARAZIONE DATI"] + rap_v + rap_a
    A["canone_totale"] = A["prezzo"] + A["spese_mese"]                  # NaN se le spese non sono indicate
    sp = A.dropna(subset=["canone_totale"]).copy()
    rap_a.append("Affitti con spese condominiali indicate (per Y2): %d su %d" % (len(sp), len(A)))
    spese_anom = sp[(sp.spese_mese > 300) | (sp.spese_mese <= 5)]
    rap_a.append("  di cui spese anomale (>300 o <=5 euro/mese), mantenute: %d (vedi out_affitti/spese_anomale.csv)" % len(spese_anom))
    scrivi_csv(spese_anom[[c for c in ["id_annuncio", "citta", "tip", "prezzo", "mq", "spese_mese", "link"] if c in spese_anom]],
               os.path.join(oa, "spese_anomale.csv"))
    scrivi_csv(V.drop(columns=["descrizione"], errors="ignore"), os.path.join(ov, "dataset_vendite_analisi.csv"))
    scrivi_csv(A.drop(columns=["descrizione"], errors="ignore"), os.path.join(oa, "dataset_affitti_analisi.csv"))
    print("\n".join(["PREPARAZIONE DATI"] + rap_v + rap_a))

    # ============================================================ DESCRITTIVA -> out_descr
    R = ["PREPARAZIONE DATI"] + rap_v + rap_a + ["", "DESCRITTIVA (prezzi in euro)"]
    test = []
    campioni = [("vendite", V, "prezzo", "prezzo di vendita (EUR)"),
                ("affitti", A, "prezzo", "canone mensile (EUR)"),
                ("affitti_con_spese", sp, "canone_totale", "canone + spese condominiali (EUR/mese)")]
    for nome, d, col, et in campioni:
        for gr, lab in (("citta", "citta"), ("tip", "tipologia")):
            t = statistiche(d, col, gr)
            scrivi_csv(t, os.path.join(od, "descr_%s_per_%s.csv" % (nome, lab)))
            R += ["", "%s - %s per %s" % (nome, et, lab), t.to_string(index=False)]
            r = test_gruppi(d, col, "citta_m" if gr == "citta" else gr, et, lab)
            if r:
                test.append(r)
            boxplot_gruppi(d, gr, col, "%s: %s" % (nome.replace("_", " ").capitalize(), et) + " per " + lab, "EUR",
                           os.path.join(od, "boxplot_%s_per_%s.png" % (nome, lab)))
    ts = statistiche(A.dropna(subset=["spese_mese"]), "spese_mese", "citta")
    scrivi_csv(ts, os.path.join(od, "descr_affitti_spese_condominiali_per_citta.csv"))
    T = pd.DataFrame(test).round(4)
    scrivi_csv(T, os.path.join(od, "test_differenze_tra_gruppi.csv"))
    R += ["", "Test di differenza tra gruppi (citta' non accorpate; gruppi con n>=3):", T.to_string(index=False),
          "(eta2 = quota di variabilita' del prezzo, in scala log, spiegata dal fattore)"]
    open(os.path.join(od, "00_riepilogo_descrittiva.txt"), "w", encoding="utf-8").write("\n".join(R))

    # ============================================================ VENDITE -> out_vendite
    RV = ["MODELLI VENDITE - Y = prezzo di vendita", "X = citta' (non accorpate) + tipologia", ""] + rap_v
    V["ly"] = np.log(V["prezzo"])
    ref_v = {"citta_m": V["citta_m"].value_counts().index[0], "tip": "Bilocale" if (V.tip == "Bilocale").any() else V.tip.value_counts().index[0]}
    m = OLS(V, "ly", ["citta_m", "tip"], ref_v)
    txt = m.riepilogo("V1. OLS: log(prezzo di vendita) ~ citta + tipologia")
    open(os.path.join(ov, "V1_ols_log_prezzo.txt"), "w", encoding="utf-8").write(txt)
    scrivi_csv(m.tabella().round(5), os.path.join(ov, "V1_ols_coefficienti.csv"))
    ncens = int((V.prezzo >= V.prezzo.max()).sum())
    t, ti = confronto_ml(V, "prezzo", ["citta_m", "tip"], "vendite (prezzo)", os.path.join(ov, "V2_confronto_modelli_cv.csv"),
                         os.path.join(ov, "V2_importanza_variabili.csv"), os.path.join(ov, "V2_confronto_e_importanza.png"), "prezzo vendita")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(np.exp(m.fitted), np.exp(m.y), s=14, color=BLU, alpha=0.7)
    lim = [min(np.exp(m.y).min(), np.exp(m.fitted).min()), max(np.exp(m.y).max(), np.exp(m.fitted).max())]
    ax.plot(lim, lim, color=GRIGIO, linewidth=1); ax.set_xlabel("previsto (EUR)"); ax.set_ylabel("osservato (EUR)")
    ax.set_title("OLS vendite: osservato vs previsto", loc="left", fontsize=11); stile(ax)
    salva_fig(fig, os.path.join(ov, "V1_osservato_vs_previsto.png"))
    RV += ["", txt, "", "Confronto modelli (validazione incrociata 5 fold x3, scala log; MAE in euro):", t.to_string(index=False),
           "", "Importanza variabili (Random Forest):", ti.to_string(index=False),
           "", "NOTA: %d vendite hanno prezzo = massimo (%d EUR): il filtro di ricerca tronca il prezzo in alto." % (ncens, V.prezzo.max())]
    open(os.path.join(ov, "00_riepilogo_vendite.txt"), "w", encoding="utf-8").write("\n".join(RV))

    # ============================================================ AFFITTI -> out_affitti
    RA = ["MODELLI AFFITTI - Y = canone mensile ; Y2 = canone + spese condominiali", "X = citta' (non accorpate) + tipologia", ""] + rap_a
    ref_a = {"citta_m": A["citta_m"].value_counts().index[0], "tip": "Bilocale" if (A.tip == "Bilocale").any() else A.tip.value_counts().index[0]}
    A["ly"] = np.log(A["prezzo"])
    sp["ly"] = np.log(sp["prezzo"]); sp["ly2"] = np.log(sp["canone_totale"])
    sp["citta_m"] = sp["citta"]
    ref_s = {"citta_m": sp["citta_m"].value_counts().index[0], "tip": ref_a["tip"]}
    mY = OLS(A, "ly", ["citta_m", "tip"], ref_a)
    mYs = OLS(sp, "ly", ["citta_m", "tip"], ref_s)
    mY2 = OLS(sp, "ly2", ["citta_m", "tip"], ref_s)
    for nome, mm, tit in (("A1_ols_log_canone", mY, "A1. OLS: log(canone) ~ citta + tipologia (tutti gli affitti)"),
                          ("A2_ols_log_canone_solo_con_spese", mYs, "A2. OLS: log(canone) ~ citta + tipologia (solo annunci con spese indicate)"),
                          ("A3_ols_log_canone_piu_spese", mY2, "A3. OLS: log(canone + spese condominiali) ~ citta + tipologia")):
        open(os.path.join(oa, nome + ".txt"), "w", encoding="utf-8").write(mm.riepilogo(tit))
        scrivi_csv(mm.tabella().round(5), os.path.join(oa, nome.split("_")[0] + "_coefficienti.csv"))
    t1, tb2, tb3 = mYs.tabella(), mY2.tabella(), None
    cmp = t1[["variabile", "effetto_%_stima", "p_value"]].merge(
        tb2[["variabile", "effetto_%_stima", "p_value"]], on="variabile", suffixes=("_Y_canone", "_Y2_canone_piu_spese"))
    cmp = cmp[cmp.variabile != "Intercetta"]
    cmp["differenza_punti_%"] = cmp["effetto_%_stima_Y2_canone_piu_spese"] - cmp["effetto_%_stima_Y_canone"]
    scrivi_csv(cmp.round(3), os.path.join(oa, "A4_confronto_Y_vs_Y2_effetti.csv"))
    riga = pd.DataFrame([{"modello": "Y canone (tutti)", "n": mY.n, "R2": mY.r2, "R2_corretto": mY.adj_r2},
                         {"modello": "Y canone (solo con spese)", "n": mYs.n, "R2": mYs.r2, "R2_corretto": mYs.adj_r2},
                         {"modello": "Y2 canone + spese", "n": mY2.n, "R2": mY2.r2, "R2_corretto": mY2.adj_r2}]).round(3)
    scrivi_csv(riga, os.path.join(oa, "A4_confronto_Y_vs_Y2_R2.csv"))
    quota = (sp["spese_mese"] / sp["canone_totale"] * 100)
    t_y, ti_y = confronto_ml(A, "prezzo", ["citta_m", "tip"], "affitti (canone)", os.path.join(oa, "A5_confronto_modelli_cv_Y.csv"),
                             os.path.join(oa, "A5_importanza_variabili_Y.csv"), os.path.join(oa, "A5_confronto_e_importanza_Y.png"), "canone")
    t_y2, ti_y2 = confronto_ml(sp, "canone_totale", ["citta_m", "tip"], "affitti (canone + spese)", os.path.join(oa, "A5_confronto_modelli_cv_Y2.csv"),
                               os.path.join(oa, "A5_importanza_variabili_Y2.csv"), os.path.join(oa, "A5_confronto_e_importanza_Y2.png"), "canone + spese")
    RA += ["", mY.riepilogo("A1. OLS: log(canone) ~ citta + tipologia (tutti gli affitti)"), "",
           mYs.riepilogo("A2. stesso modello sui soli annunci con spese indicate"), "",
           mY2.riepilogo("A3. OLS: log(canone + spese condominiali) ~ citta + tipologia"), "",
           "A4. Confronto Y (canone) e Y2 (canone + spese) sullo STESSO campione (effetti % rispetto al riferimento):",
           cmp.round(2).to_string(index=False), "", riga.to_string(index=False),
           "Peso delle spese sul costo totale: mediana %.1f%% | media %.1f%% (n=%d)" % (quota.median(), quota.mean(), len(sp)),
           "", "A5. Confronto modelli in validazione incrociata (scala log; MAE in euro)", "Y = canone:", t_y.to_string(index=False),
           "Y2 = canone + spese:", t_y2.to_string(index=False), "", "Importanza variabili (Random Forest) - Y:", ti_y.to_string(index=False),
           "Importanza variabili (Random Forest) - Y2:", ti_y2.to_string(index=False)]
    open(os.path.join(oa, "00_riepilogo_affitti.txt"), "w", encoding="utf-8").write("\n".join(RA))
    print("\n".join(RV)); print("\n" + "\n".join(RA))
    print("\nRisultati salvati in:\n  %s\n  %s\n  %s" % (od, ov, oa))
    return 0


if __name__ == "__main__":
    sys.exit(main())
