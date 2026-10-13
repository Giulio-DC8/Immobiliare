# -*- coding: utf-8 -*-
"""
GDC Immobiliare - analisi statistica completa degli annunci (modelli A-F) in UN solo script.

Input : output.csv (vendite + affitti nello stesso file) nella cartella del progetto,
        oppure, se manca, annunci_vendita_spese.csv + annunci_affitto_spese.csv.
Output: cartella analisi_modelli/ (tabelle CSV, grafici PNG, 00_RIEPILOGO.txt, dataset puliti).

Modelli
  A  Descrittivo: statistiche per citta' e tipologia + test (Kruskal-Wallis, ANOVA) + boxplot
  B  Vendite:  OLS  log(prezzo/mq) ~ citta + tipologia + log(mq)            (errori robusti HC3)
  C  Affitti:  OLS  log(canone)    ~ citta + tipologia + log(mq)            -> canone stimato per ogni vendita
  D  Rendimento per annuncio: canone stimato (C) x 12 / prezzo; lordo e netto; classifica; OLS sul rendimento
  E  Machine learning (Random Forest, Gradient Boosting) vs Ridge, con validazione incrociata
  F  Vendite:  OLS  log(prezzo) ~ citta + tipologia                         (come richiesto, senza mq)

Uso (Anaconda Prompt, nella cartella del progetto):
  python analisi_modelli.py
  python analisi_modelli.py --input output.csv --out analisi_modelli --min-citta 5 --max-mq-affitto 100
Dipendenze: pandas, numpy, scipy, scikit-learn, matplotlib (tutte incluse in Anaconda). Nessun accesso a internet.
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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
AQUI = os.path.dirname(os.path.abspath(__file__))

# Correzioni dei nomi citta' ricavati dal nome del file (aggiungi qui eventuali altri errori)
CORREZIONI = {"verese": "Varese", "mestra": "Mestre", "pesarourbino": "Pesaro", "pesaro urbino": "Pesaro",
              "san raffaele": "Milano"}
COL_RIF = ["rif", "rif_openai", "rif_gemini", "rif_grok"]
BLU, GRIGIO, FONDO = "#2a78d6", "#8a8f98", "#fcfcfb"


# ----------------------------------------------------------------------------- utilita'
def leggi_csv(path):
    txt = None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            with open(path, encoding=enc) as fh:
                txt = fh.read()
            break
        except UnicodeDecodeError:
            continue
    prima = txt.split("\n", 1)[0]
    sep = ";" if prima.count(";") > prima.count(",") else ","
    df = pd.read_csv(io.StringIO(txt), sep=sep, dtype=str, keep_default_na=False)
    return df, sep


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
    if t.startswith("monolocale"):
        return "Monolocale"
    if t.startswith("bilocale"):
        return "Bilocale"
    if t.startswith("trilocale"):
        return "Trilocale"
    if t.startswith("quadrilocale") or t.startswith("pentalocale") or t.startswith("plurilocale"):
        return "Quadrilocale+"
    if t.startswith("attico") or t.startswith("casa") or t.startswith("villa") or "due piani" in t:
        return "Altro (attico/casa)"
    if pd.notna(locali):                       # "Appartamento" generico: uso il numero di locali
        return {1: "Monolocale", 2: "Bilocale", 3: "Trilocale"}.get(int(locali), "Quadrilocale+")
    return "Altro (attico/casa)"


def raggruppa(serie, min_n):
    n = serie.value_counts()
    rare = set(n[n < min_n].index)
    return serie.where(~serie.isin(rare), "Altre")


def outlier_mad(x, soglia=4.0):
    l = np.log(x)
    mad = np.median(np.abs(l - np.median(l))) * 1.4826
    return np.abs(l - np.median(l)) > soglia * mad if mad > 0 else pd.Series(False, index=x.index)


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


# ----------------------------------------------------------------------------- OLS (numpy, errori robusti HC3)
class OLS:
    """Regressione lineare con variabile risposta y, colonne numeriche e categoriali (dummy con livello di riferimento)."""

    def __init__(self, df, y, cat, num=(), ref=None):
        self.cat, self.num, self.ref = list(cat), list(num), ref or {}
        d = df.dropna(subset=[y] + self.cat + self.num).copy()
        X, self.nomi = self._design(d)
        yv = d[y].to_numpy(float)
        self.n, self.k = X.shape
        self.beta = np.linalg.lstsq(X, yv, rcond=None)[0]
        self.fitted = X @ self.beta
        self.resid = yv - self.fitted
        self.rss = float(self.resid @ self.resid)
        self.tss = float(((yv - yv.mean()) ** 2).sum())
        self.r2 = 1 - self.rss / self.tss
        self.df_res = self.n - self.k
        self.adj_r2 = 1 - (1 - self.r2) * (self.n - 1) / self.df_res
        XtXi = np.linalg.pinv(X.T @ X)
        h = np.einsum("ij,jk,ik->i", X, XtXi, X)
        w = (self.resid / (1 - np.clip(h, 0, 0.999))) ** 2
        self.V = XtXi @ (X.T * w) @ X @ XtXi
        self.se = np.sqrt(np.diag(self.V))
        self.X, self.y, self.data, self.yname = X, yv, d, y
        self.X_cols = self.nomi
        self.sigma2 = self.rss / self.df_res
        self.F = ((self.tss - self.rss) / (self.k - 1)) / self.sigma2 if self.k > 1 else np.nan
        self.pF = 1 - stats.f.cdf(self.F, self.k - 1, self.df_res) if self.k > 1 else np.nan

    def _design(self, d, livelli=None):
        cols, nomi = [np.ones(len(d))], ["Intercetta"]
        self.livelli = {} if livelli is None else livelli
        for c in self.cat:
            if livelli is None:
                lv = d[c].value_counts().index.tolist()
                rif = self.ref.get(c, lv[0])
                self.livelli[c] = (rif, [x for x in sorted(set(d[c])) if x != rif])
            rif, altri = self.livelli[c]
            for x in altri:
                cols.append((d[c] == x).to_numpy(float)); nomi.append("%s = %s (vs %s)" % (c, x, rif))
        for c in self.num:
            cols.append(d[c].to_numpy(float)); nomi.append(c)
        return np.column_stack(cols), nomi

    def predici(self, d):
        X, _ = self._design(d, livelli=self.livelli)
        return X @ self.beta

    def tabella(self, log_y=True):
        t = self.beta / self.se
        p = 2 * (1 - stats.t.cdf(np.abs(t), self.df_res))
        q = stats.t.ppf(0.975, self.df_res)
        out = pd.DataFrame({"variabile": self.nomi, "coef": self.beta, "errore_std_HC3": self.se, "t": t,
                            "p_value": p, "ci95_inf": self.beta - q * self.se, "ci95_sup": self.beta + q * self.se})
        if log_y:
            for c in ("coef", "ci95_inf", "ci95_sup"):
                out["effetto_%_" + c] = (np.exp(out[c]) - 1) * 100
            out.loc[out.variabile == "Intercetta", [c for c in out.columns if c.startswith("effetto")]] = np.nan
        return out

    def test_fattori(self):
        """Test F parziale classico: ogni fattore (citta', tipologia, ...) e' significativo?"""
        righe = []
        for f in self.cat + self.num:
            sotto = OLS(self.data, self.yname, [c for c in self.cat if c != f], [c for c in self.num if c != f], self.ref)
            dfn = self.k - sotto.k
            if dfn <= 0:
                continue
            F = ((sotto.rss - self.rss) / dfn) / self.sigma2
            righe.append({"fattore": f, "gradi_liberta": dfn, "F": F, "p_value": 1 - stats.f.cdf(F, dfn, self.df_res)})
        return pd.DataFrame(righe)

    def riepilogo(self, titolo, log_y=True):
        r = ["=" * 78, titolo, "=" * 78,
             "osservazioni %d | parametri %d | R2 %.3f | R2 corretto %.3f | F %.2f (p=%.4g)" %
             (self.n, self.k, self.r2, self.adj_r2, self.F, self.pF),
             "errore standard residuo (scala log) %.3f" % np.sqrt(self.sigma2), ""]
        tb = self.tabella(log_y)
        cols = ["variabile", "coef", "p_value"] + (["effetto_%_coef"] if log_y else [])
        r.append(tb[cols].round(4).to_string(index=False))
        r += ["", "Significativita' dei fattori (test F parziale):", self.test_fattori().round(4).to_string(index=False)]
        return "\n".join(r)


def cv_modello(pipe, X, y, reps=3, k=5):
    r2s, maes = [], []
    for s in range(reps):
        p = cross_val_predict(pipe, X, y, cv=KFold(k, shuffle=True, random_state=42 + s))
        r2s.append(r2_score(y, p)); maes.append(mean_absolute_error(np.exp(y), np.exp(p)))
    return float(np.mean(r2s)), float(np.std(r2s)), float(np.mean(maes))


def prepro(cat, num):
    return ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat), ("n", "passthrough", num)] if num
                             else [("c", OneHotEncoder(handle_unknown="ignore"), cat)])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="output.csv")
    ap.add_argument("--out", default="analisi_modelli", help="cartella dei risultati")
    ap.add_argument("--min-citta", type=int, default=5, help="citta' con meno annunci vanno in 'Altre'")
    ap.add_argument("--max-mq-affitto", type=float, default=100.0, help="massimo mq per gli affitti usati nel modello C")
    ap.add_argument("--max-mq-vendita", type=float, default=None, help="massimo mq per le vendite (default: nessun taglio)")
    ap.add_argument("--quota-costi", type=float, default=0.15,
                    help="quota del canone assorbita da vacancy/manutenzione/imposte per il rendimento netto (ASSUNZIONE)")
    ap.add_argument("--quota-spese", type=float, default=1.0,
                    help="quota delle spese condominiali a carico del proprietario (1 = tutte, 0 = nessuna) (ASSUNZIONE)")
    ap.add_argument("--senza-filtro-rif", action="store_true", help="non escludere annunci con rif/rif_* = NO")
    a = ap.parse_args()

    out = a.out if os.path.isabs(a.out) else os.path.join(AQUI, a.out)
    os.makedirs(out, exist_ok=True)
    R = []                                           # righe del riepilogo
    log = lambda s="": (R.append(s), print(s))

    # ---------------------------------------------------------------- 1. dati
    f = a.input if os.path.isabs(a.input) else os.path.join(AQUI, a.input)
    if os.path.exists(f):
        df, sep = leggi_csv(f)
    else:
        parti = []
        for op in ("vendita", "affitto"):
            p = os.path.join(AQUI, "annunci_%s_spese.csv" % op)
            if os.path.exists(p):
                d, sep = leggi_csv(p); parti.append(d)
        if not parti:
            print("Nessun file di input trovato (%s)." % a.input); return 1
        df = pd.concat(parti, ignore_index=True)
    n_iniziale = len(df)
    for c in ("prezzo", "mq", "locali", "spese_mese", "spese_anno", "prezzo_mq"):
        if c in df:
            df[c] = numerico(df[c], sep)
    df["operazione"] = df["operazione"].str.strip().str.lower()
    df["citta"] = [citta_da_file(fs, ca) for fs, ca in zip(df.get("file_sorgente", ""), df.get("citta_area", ""))]
    df["tip"] = [tipologia_acc(t, l) for t, l in zip(df["tipologia"], df["locali"])]
    log("GDC IMMOBILIARE - ANALISI MODELLI A-F")
    log("File: %s | righe lette: %d" % (os.path.basename(f), n_iniziale))

    scartati = {}
    ok = (df["prezzo"] > 0) & (df["mq"] > 0)
    scartati["prezzo o mq mancanti"] = int((~ok).sum()); df = df[ok].copy()
    if not a.senza_filtro_rif:
        presenti = [c for c in COL_RIF if c in df]
        if presenti:
            m = pd.Series(True, index=df.index)
            for c in presenti:
                m &= ~df[c].str.upper().eq("NO")
            scartati["rif/rif_* = NO (non immobile): " + ",".join(presenti)] = int((~m).sum()); df = df[m].copy()
    df["prezzo_mq"] = df["prezzo"] / df["mq"]
    v = df[df.operazione == "vendita"].copy()
    af = df[df.operazione == "affitto"].copy()
    if a.max_mq_vendita:
        scartati["vendite oltre %d mq" % a.max_mq_vendita] = int((v.mq > a.max_mq_vendita).sum()); v = v[v.mq <= a.max_mq_vendita]
    sc_aff = int((af.mq > a.max_mq_affitto).sum()); af = af[af.mq <= a.max_mq_affitto]
    scartati["affitti oltre %d mq (solo per C)" % a.max_mq_affitto] = sc_aff
    o1, o2 = outlier_mad(v["prezzo_mq"]), outlier_mad(af["prezzo_mq"])
    scartati["outlier vendite (MAD>4, prezzo/mq)"] = int(o1.sum()); v = v[~o1]
    scartati["outlier affitti (MAD>4, canone/mq)"] = int(o2.sum()); af = af[~o2]
    log("Esclusioni: " + "; ".join("%s=%d" % kv for kv in scartati.items()))
    log("Campione finale: %d vendite, %d affitti" % (len(v), len(af)))
    if len(v) < 30 or len(af) < 30:
        log("ATTENZIONE: campione molto piccolo, i risultati dei modelli sono poco affidabili.")
    ncens = int((v.prezzo >= v.prezzo.max()).sum())
    log("Nota: %d vendite hanno prezzo = massimo (%d): il filtro di ricerca tronca il prezzo in alto -> per questo "
        "i modelli B/E usano prezzo/mq; F usa il prezzo come richiesto." % (ncens, v.prezzo.max()))

    v["citta_m"] = raggruppa(v["citta"], a.min_citta)
    af["citta_m"] = raggruppa(af["citta"], a.min_citta)
    v["lmq"], af["lmq"] = np.log(v["mq"]), np.log(af["mq"])
    v["ly_mq"], v["ly_prezzo"] = np.log(v["prezzo_mq"]), np.log(v["prezzo"])
    af["ly_canone"] = np.log(af["prezzo"])
    ref_v = {"citta_m": v["citta_m"].value_counts().index[0], "tip": "Bilocale" if (v.tip == "Bilocale").any() else v.tip.value_counts().index[0]}
    ref_a = {"citta_m": af["citta_m"].value_counts().index[0], "tip": "Bilocale" if (af.tip == "Bilocale").any() else af.tip.value_counts().index[0]}
    v.to_csv(os.path.join(out, "dataset_pulito_vendite.csv"), sep=";", index=False, encoding="utf-8-sig")
    af.to_csv(os.path.join(out, "dataset_pulito_affitti.csv"), sep=";", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- A. descrittivo
    log("\n" + "=" * 78 + "\nA. DESCRITTIVO\n" + "=" * 78)
    def desc(d, col, gruppo):
        g = d.groupby(gruppo)[col]
        t = g.agg(n="size", media="mean", mediana="median", dev_std="std",
                  p25=lambda s: s.quantile(.25), p75=lambda s: s.quantile(.75), min="min", max="max")
        return t.sort_values("mediana", ascending=False).round(1).reset_index()
    for nome, d in (("vendite", v), ("affitti", af)):
        for gr in ("citta", "tip"):
            for col, et in (("prezzo", "prezzo" if nome == "vendite" else "canone"), ("prezzo_mq", "prezzo_mq" if nome == "vendite" else "canone_mq")):
                desc(d, col, gr).to_csv(os.path.join(out, "A_%s_per_%s_%s.csv" % (nome, "citta" if gr == "citta" else "tipologia", et)),
                                        sep=";", index=False, encoding="utf-8-sig")
    righe = []
    for nome, d in (("vendite", v), ("affitti", af)):
        for gr, et in (("citta_m", "citta"), ("tip", "tipologia")):
            gruppi = [g["prezzo_mq"].to_numpy() for _, g in d.groupby(gr) if len(g) >= 3]
            if len(gruppi) < 2:
                continue
            kw = stats.kruskal(*gruppi)
            lg = [np.log(x) for x in gruppi]
            an = stats.f_oneway(*lg)
            tutti = np.concatenate(lg)
            ssb = sum(len(x) * (x.mean() - tutti.mean()) ** 2 for x in lg)
            righe.append({"campione": nome, "fattore": et, "gruppi": len(gruppi), "Kruskal_H": kw.statistic, "Kruskal_p": kw.pvalue,
                          "ANOVA_F(log)": an.statistic, "ANOVA_p": an.pvalue, "eta2": ssb / ((tutti - tutti.mean()) ** 2).sum()})
    tA = pd.DataFrame(righe)
    tA.round(4).to_csv(os.path.join(out, "A_test_differenze_gruppi.csv"), sep=";", index=False, encoding="utf-8-sig")
    log(tA.round(4).to_string(index=False))
    log("(eta2 = quota di variabilita' di prezzo/mq spiegata dal fattore)")
    log("\nVendite - prezzo/mq per citta' (mediana):\n" + desc(v, "prezzo_mq", "citta")[["citta", "n", "mediana", "p25", "p75"]].to_string(index=False))

    def boxplot_gruppi(d, gr, col, titolo, ylabel, path, min_n=3):
        ordine = d.groupby(gr)[col].median().sort_values(ascending=False)
        ordine = ordine[d.groupby(gr)[col].size()[ordine.index] >= min_n]
        fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(ordine) + 2), 5))
        dati = [d.loc[d[gr] == c, col] for c in ordine.index]
        bp = ax.boxplot(dati, patch_artist=True, showfliers=True, medianprops=dict(color="#12326a", linewidth=1.6),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=GRIGIO, markeredgecolor=GRIGIO))
        for b in bp["boxes"]:
            b.set(facecolor=BLU, alpha=0.55, edgecolor=BLU)
        ax.set_xticks(range(1, len(ordine) + 1))
        ax.set_xticklabels(["%s\n(n=%d)" % (c, len(x)) for c, x in zip(ordine.index, dati)], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(titolo, loc="left", fontsize=11); stile(ax)
        salva_fig(fig, path)
    boxplot_gruppi(v, "citta", "prezzo_mq", "Vendite: prezzo al m2 per citta'", "EUR / m2", os.path.join(out, "A_boxplot_vendite_citta.png"))
    boxplot_gruppi(af, "citta", "prezzo_mq", "Affitti: canone al m2 per citta'", "EUR / m2 / mese", os.path.join(out, "A_boxplot_affitti_citta.png"))
    boxplot_gruppi(v, "tip", "prezzo_mq", "Vendite: prezzo al m2 per tipologia", "EUR / m2", os.path.join(out, "A_boxplot_vendite_tipologia.png"), 2)
    boxplot_gruppi(af, "tip", "prezzo_mq", "Affitti: canone al m2 per tipologia", "EUR / m2 / mese", os.path.join(out, "A_boxplot_affitti_tipologia.png"), 2)

    # ---------------------------------------------------------------- B. OLS vendite log(prezzo/mq)
    log("\n" + "=" * 78 + "\nB. OLS VENDITE: log(prezzo/mq) ~ citta + tipologia + log(mq)\n" + "=" * 78)
    mB = OLS(v, "ly_mq", ["citta_m", "tip"], ["lmq"], ref_v)
    open(os.path.join(out, "B_modello_vendite_prezzo_mq.txt"), "w", encoding="utf-8").write(mB.riepilogo("B. log(prezzo/mq) ~ citta + tipologia + log(mq)"))
    mB.tabella().round(5).to_csv(os.path.join(out, "B_coefficienti.csv"), sep=";", index=False, encoding="utf-8-sig")
    log("R2 %.3f | R2 corretto %.3f | n=%d | riferimento: %s" % (mB.r2, mB.adj_r2, mB.n, ref_v))
    log(mB.test_fattori().round(4).to_string(index=False))
    log("Effetto % sul prezzo/mq (a parita' del resto):\n" +
        mB.tabella()[["variabile", "effetto_%_coef", "p_value"]].dropna().round(3).to_string(index=False))
    Xb = v[["citta_m", "tip", "lmq"]]
    pipe_b = make_pipeline(prepro(["citta_m", "tip"], ["lmq"]), RidgeCV(alphas=np.logspace(-2, 2, 25)))
    cvB = cv_modello(pipe_b, Xb, v["ly_mq"].to_numpy())
    log("Validazione incrociata (Ridge, 5 fold x3): R2 CV %.3f (+/- %.3f) | MAE %.0f EUR/m2" % cvB)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(np.exp(mB.fitted), np.exp(mB.y), s=14, color=BLU, alpha=0.7)
    lim = [min(np.exp(mB.y).min(), np.exp(mB.fitted).min()), max(np.exp(mB.y).max(), np.exp(mB.fitted).max())]
    ax.plot(lim, lim, color=GRIGIO, linewidth=1); ax.set_xlabel("previsto EUR/m2"); ax.set_ylabel("osservato EUR/m2")
    ax.set_title("Modello B: osservato vs previsto", loc="left", fontsize=11); stile(ax)
    salva_fig(fig, os.path.join(out, "B_osservato_vs_previsto.png"))

    # ---------------------------------------------------------------- C. OLS affitti
    log("\n" + "=" * 78 + "\nC. OLS AFFITTI: log(canone) ~ citta + tipologia + log(mq)\n" + "=" * 78)
    mC = OLS(af, "ly_canone", ["citta_m", "tip"], ["lmq"], ref_a)
    open(os.path.join(out, "C_modello_affitti_canone.txt"), "w", encoding="utf-8").write(mC.riepilogo("C. log(canone mensile) ~ citta + tipologia + log(mq)"))
    mC.tabella().round(5).to_csv(os.path.join(out, "C_coefficienti.csv"), sep=";", index=False, encoding="utf-8-sig")
    log("R2 %.3f | R2 corretto %.3f | n=%d | riferimento: %s" % (mC.r2, mC.adj_r2, mC.n, ref_a))
    log(mC.test_fattori().round(4).to_string(index=False))
    pipe_c = make_pipeline(prepro(["citta_m", "tip"], ["lmq"]), RidgeCV(alphas=np.logspace(-2, 2, 25)))
    cvC = cv_modello(pipe_c, af[["citta_m", "tip", "lmq"]], af["ly_canone"].to_numpy())
    log("Validazione incrociata (Ridge, 5 fold x3): R2 CV %.3f (+/- %.3f) | MAE %.0f EUR/mese" % cvC)
    smear = float(np.mean(np.exp(mC.resid)))                      # correzione di Duan (ritorno dalla scala log)

    # ---------------------------------------------------------------- D. rendimento
    log("\n" + "=" * 78 + "\nD. RENDIMENTO PER ANNUNCIO (canone stimato dal modello C)\n" + "=" * 78)
    d = v.copy()
    livelli_citta = set(mC.livelli["citta_m"][1]) | {mC.livelli["citta_m"][0]}
    d["citta_c"] = d["citta"].where(d["citta"].isin(livelli_citta), "Altre")
    if "Altre" not in livelli_citta:
        d.loc[d["citta_c"] == "Altre", "citta_c"] = np.nan
    d["citta_m_orig"] = d["citta_m"]; d["citta_m"] = d["citta_c"]
    tip_ok = set(mC.livelli["tip"][1]) | {mC.livelli["tip"][0]}
    d.loc[~d["tip"].isin(tip_ok), "tip"] = np.nan
    stima = pd.Series(np.nan, index=d.index)
    mask = d[["citta_m", "tip", "lmq"]].notna().all(axis=1)
    stima[mask] = np.exp(mC.predici(d[mask])) * smear
    d["citta_m"] = d["citta_m_orig"]; d["tip"] = v["tip"]
    d["canone_stimato"] = stima
    if "spese_anno" not in d or d["spese_anno"].isna().all():
        d["spese_anno"] = d["spese_mese"] * 12 if "spese_mese" in d else np.nan
    med_cit = d.groupby("citta")["spese_anno"].transform("median")
    d["spese_imputate"] = d["spese_anno"].isna()
    d["spese_anno_uso"] = d["spese_anno"].fillna(med_cit).fillna(d["spese_anno"].median()).fillna(0)
    d["rend_lordo_%"] = d["canone_stimato"] * 12 / d["prezzo"] * 100
    d["rend_netto_%"] = (d["canone_stimato"] * 12 * (1 - a.quota_costi) - d["spese_anno_uso"] * a.quota_spese) / d["prezzo"] * 100
    cols_d = [c for c in ["id_annuncio", "citta", "tip", "titolo", "prezzo", "mq", "prezzo_mq", "canone_stimato", "spese_anno_uso",
                          "spese_imputate", "rend_lordo_%", "rend_netto_%", "link"] if c in d]
    D = d[cols_d].sort_values("rend_netto_%", ascending=False)
    D.round(2).to_csv(os.path.join(out, "D_rendimento_annunci.csv"), sep=";", index=False, encoding="utf-8-sig")
    D.head(30).round(2).to_csv(os.path.join(out, "D_top30_annunci.csv"), sep=";", index=False, encoding="utf-8-sig")
    agg = lambda g: d.groupby(g).agg(n=("prezzo", "size"), prezzo_mediano=("prezzo", "median"), canone_stimato_mediano=("canone_stimato", "median"),
                                      rend_lordo_mediano=("rend_lordo_%", "median"), rend_netto_mediano=("rend_netto_%", "median"),
                                      rend_netto_p25=("rend_netto_%", lambda s: s.quantile(.25)),
                                      rend_netto_p75=("rend_netto_%", lambda s: s.quantile(.75))).sort_values("rend_netto_mediano", ascending=False).round(2).reset_index()
    dc, dt = agg("citta"), agg("tip")
    dc.to_csv(os.path.join(out, "D_rendimento_per_citta.csv"), sep=";", index=False, encoding="utf-8-sig")
    dt.to_csv(os.path.join(out, "D_rendimento_per_tipologia.csv"), sep=";", index=False, encoding="utf-8-sig")
    log("Assunzioni: costi = %.0f%% del canone (vacancy/manutenzione/imposte); spese condominiali a carico del proprietario = %.0f%%. "
        "Modificabili con --quota-costi e --quota-spese." % (a.quota_costi * 100, a.quota_spese * 100))
    log("Annunci con canone stimato: %d su %d | spese condominiali imputate (mediana citta'): %d" %
        (d["canone_stimato"].notna().sum(), len(d), int(d["spese_imputate"].sum())))
    log("\nRendimento per citta' (mediane, %):\n" + dc.to_string(index=False))
    log("\nRendimento per tipologia:\n" + dt.to_string(index=False))
    dd = d.dropna(subset=["rend_lordo_%"]).copy()
    dd["ly_rend"] = np.log(dd["rend_lordo_%"])
    dd["citta_r"] = raggruppa(dd["citta"], a.min_citta)
    mD = OLS(dd, "ly_rend", ["citta_r", "tip"], ["lmq"], {"citta_r": dd["citta_r"].value_counts().index[0], "tip": ref_v["tip"]})
    open(os.path.join(out, "D_modello_rendimento.txt"), "w", encoding="utf-8").write(mD.riepilogo("D. log(rendimento lordo) ~ citta + tipologia + log(mq)"))
    log("Modello del rendimento: R2 %.3f | n=%d (vedi D_modello_rendimento.txt)" % (mD.r2, mD.n))
    top = dc[dc["n"] >= 3].sort_values("rend_netto_mediano")
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.4 * len(top) + 1.5)))
    ax.barh(top["citta"], top["rend_lordo_mediano"], color=BLU, alpha=0.35, label="lordo")
    ax.barh(top["citta"], top["rend_netto_mediano"], color=BLU, label="netto")
    for i, (c, n) in enumerate(zip(top["citta"], top["n"])):
        ax.text(0.15, i, "n=%d" % n, va="center", fontsize=7, color="white")
    ax.set_xlabel("rendimento annuo mediano (%)"); ax.set_title("Rendimento stimato per citta'", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8); stile(ax); ax.grid(axis="x", color="#e6e6e3"); ax.grid(axis="y", visible=False)
    salva_fig(fig, os.path.join(out, "D_rendimento_citta.png"))

    # ---------------------------------------------------------------- E. machine learning
    log("\n" + "=" * 78 + "\nE. MACHINE LEARNING (validazione incrociata 5 fold x3) - target: log(prezzo/mq)\n" + "=" * 78)
    feat_cat = ["citta_m", "tip"] + [c for c in ("ascensore", "box") if c in v and v[c].nunique() > 1]
    feat_num = ["lmq"] + (["locali"] if "locali" in v and v["locali"].notna().any() else [])
    XE = v[feat_cat + feat_num].copy()
    for c in feat_cat:
        XE[c] = XE[c].astype(str)
    if "locali" in feat_num:
        XE["locali"] = XE["locali"].fillna(XE["locali"].median())
    yE = v["ly_mq"].to_numpy()
    modelli = {
        "Media (nessun predittore)": make_pipeline(prepro(feat_cat, feat_num), DummyRegressor()),
        "Ridge (lineare)": make_pipeline(prepro(feat_cat, feat_num), RidgeCV(alphas=np.logspace(-2, 2, 25))),
        "Random Forest": make_pipeline(prepro(feat_cat, feat_num), RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=42, n_jobs=-1)),
        "Gradient Boosting": make_pipeline(prepro(feat_cat, feat_num), GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8, random_state=42)),
    }
    righe = []
    for nome, p in modelli.items():
        r2m, r2s, mae = cv_modello(p, XE, yE)
        righe.append({"modello": nome, "R2_CV": r2m, "R2_CV_dev_std": r2s, "MAE_EUR_mq": mae})
    tE = pd.DataFrame(righe).round(3)
    tE.to_csv(os.path.join(out, "E_confronto_modelli.csv"), sep=";", index=False, encoding="utf-8-sig")
    log("Variabili usate: %s\n%s" % (feat_cat + feat_num, tE.to_string(index=False)))
    # importanza delle variabili (permutazione, su dati fuori campione)
    from sklearn.inspection import permutation_importance
    imp = pd.DataFrame(0.0, index=feat_cat + feat_num, columns=["importanza"])
    for tr, te in KFold(5, shuffle=True, random_state=1).split(XE):
        rf = make_pipeline(prepro(feat_cat, feat_num), RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1))
        rf.fit(XE.iloc[tr], yE[tr])
        pi = permutation_importance(rf, XE.iloc[te], yE[te], n_repeats=10, random_state=0, scoring="r2")
        imp["importanza"] += pi.importances_mean / 5
    imp = imp.sort_values("importanza", ascending=False).round(4).reset_index().rename(columns={"index": "variabile"})
    imp.to_csv(os.path.join(out, "E_importanza_variabili.csv"), sep=";", index=False, encoding="utf-8-sig")
    log("Importanza variabili (calo di R2 permutando la variabile, Random Forest):\n" + imp.to_string(index=False))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.barh(tE["modello"][::-1], tE["R2_CV"][::-1], color=BLU); ax1.set_xlabel("R2 in validazione incrociata")
    ax1.set_title("Confronto modelli", loc="left", fontsize=11); stile(ax1)
    ax2.barh(imp["variabile"][::-1], imp["importanza"][::-1], color=BLU); ax2.set_xlabel("importanza (calo di R2)")
    ax2.set_title("Variabili piu' importanti", loc="left", fontsize=11); stile(ax2)
    salva_fig(fig, os.path.join(out, "E_confronto_e_importanza.png"))

    # ---------------------------------------------------------------- F. OLS vendite log(prezzo) ~ citta + tipologia
    log("\n" + "=" * 78 + "\nF. OLS VENDITE: log(prezzo) ~ citta + tipologia\n" + "=" * 78)
    mF = OLS(v, "ly_prezzo", ["citta_m", "tip"], [], ref_v)
    open(os.path.join(out, "F_modello_vendite_log_prezzo.txt"), "w", encoding="utf-8").write(mF.riepilogo("F. log(prezzo di vendita) ~ citta + tipologia"))
    mF.tabella().round(5).to_csv(os.path.join(out, "F_coefficienti.csv"), sep=";", index=False, encoding="utf-8-sig")
    log("R2 %.3f | R2 corretto %.3f | n=%d | riferimento: %s" % (mF.r2, mF.adj_r2, mF.n, ref_v))
    log(mF.test_fattori().round(4).to_string(index=False))
    log("Effetto % sul prezzo (a parita' dell'altra variabile):\n" +
        mF.tabella()[["variabile", "effetto_%_coef", "p_value"]].dropna().round(3).to_string(index=False))
    pipe_f = make_pipeline(prepro(["citta_m", "tip"], []), RidgeCV(alphas=np.logspace(-2, 2, 25)))
    cvF = cv_modello(pipe_f, v[["citta_m", "tip"]], v["ly_prezzo"].to_numpy())
    log("Validazione incrociata (Ridge): R2 CV %.3f (+/- %.3f) | MAE %.0f EUR" % cvF)
    log("Confronto F vs B: R2 corretto F = %.3f (solo citta' e tipologia) contro B = %.3f (aggiunge i mq)." % (mF.adj_r2, mB.adj_r2))

    log("\nFile salvati in: %s" % out)
    open(os.path.join(out, "00_RIEPILOGO.txt"), "w", encoding="utf-8").write("\n".join(R))
    return 0


if __name__ == "__main__":
    sys.exit(main())
