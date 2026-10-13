#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kriging_citta.py  -  analisi geospaziale (regression kriging) di prezzi di vendita e canoni di affitto.

INPUT (solo questi due file, nella cartella analisi_modelli):
    output_vendite_GEO.csv
    output_affitti_GEO.csv

PER OGNI CITTA' E OPERAZIONE
    1. modello log(prezzo) ~ log(mq) + tipologia   (solo log(mq) se si usano soli bilocali)
    2. test di Moran sui residui (c'e' struttura spaziale?)
    3. se n >= --n-min e p <= --p-max: variogramma + kriging ordinario dei residui
    4. cross-validation leave-one-out: kriging vs sola regressione vs media cittadina
    5. stima per il bilocale tipo (--mq-tipo m2) su griglia, con incertezza
    6. dove vendite e affitti sono entrambi "kriging utile": mappa del rendimento

OUTPUT in analisi_modelli/out_kriging/ :
    kriging_riepilogo.csv, kriging_punti.csv, kriging_rendimento_riepilogo.csv,
    griglie/*.csv, png/*.png, mappe_kriging.html

DIPENDENZE: numpy, pandas, scipy, matplotlib (gia' presenti in Anaconda).
Opzionale: contextily (sfondo cartografico nei PNG):  pip install contextily
La mappa HTML richiede internet solo per caricare Leaflet e lo sfondo.
"""
import argparse, base64, io, json, math, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

try:
    import contextily as cx
except Exception:
    cx = None

warnings.filterwarnings("ignore")
Z80 = 1.2816  # quantile normale per intervallo 80%
OPS = ("vendita", "affitto")
UNITA = {"vendita": "€ (bilocale tipo)", "affitto": "€/mese (bilocale tipo)"}


# ----------------------------------------------------------------------------- lettura
def leggi(path):
    raw = Path(path).read_bytes()
    try:
        txt = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        txt = raw.decode("cp1252")
    testa = txt[:5000]
    sep = ";" if testa.count(";") > testa.count(",") else ","
    return pd.read_csv(io.StringIO(txt), sep=sep, dtype=str, keep_default_na=False)


def num(s):
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def prepara(df, nome):
    d = df.copy()
    col = "città" if "città" in d.columns else None
    if col is None:
        sys.exit("ERRORE: nel file %s manca la colonna 'città'" % nome)
    d["citta"] = d[col].astype(str).str.strip().str.lower().str.replace(" ", "", regex=False)
    for c in ("prezzo", "mq", "latitudine", "longitudine"):
        d[c] = num(d[c])
    n0 = len(d)
    ok = ((d["id_annuncio"].astype(str).str.strip() != "") & d["latitudine"].notna() & d["longitudine"].notna()
          & (d["prezzo"] > 0) & (d["mq"] > 0) & (d["citta"] != ""))
    d = d[ok].copy()
    d["tipologia"] = d["tipologia"].astype(str).str.strip()
    print("  %-26s righe %4d -> valide %4d (scartate %d: senza id/coordinate/prezzo/mq/città)"
          % (nome, n0, len(d), n0 - len(d)))
    return d.reset_index(drop=True)


# ----------------------------------------------------------------------------- geometria
def proietta(lat, lon, lat0, lon0):
    x = (np.asarray(lon) - lon0) * 111320.0 * math.cos(math.radians(lat0))
    y = (np.asarray(lat) - lat0) * 110540.0
    return np.c_[x, y]


def xy_a_ll(x, y, lat0, lon0):
    return lat0 + np.asarray(y) / 110540.0, lon0 + np.asarray(x) / (111320.0 * math.cos(math.radians(lat0)))


def merc(lon, lat):
    R = 6378137.0
    return R * np.radians(lon), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def separa_duplicati(xy, jitter, rng):
    """Posizioni identiche -> piccolo spostamento casuale (solo dalla seconda occorrenza)."""
    xy = xy.copy()
    visti = {}
    n_mossi = 0
    for i in range(len(xy)):
        k = (round(xy[i, 0], 1), round(xy[i, 1], 1))
        if k in visti:
            ang, rr = rng.uniform(0, 2 * math.pi), jitter * math.sqrt(rng.uniform(0.05, 1))
            xy[i] += [rr * math.cos(ang), rr * math.sin(ang)]
            n_mossi += 1
        else:
            visti[k] = i
    return xy, n_mossi


# ----------------------------------------------------------------------------- regressione
def disegno(g, mq_tipo, min_liv=5, rif="Bilocale"):
    """Matrice X: costante, log(mq), dummy di tipologia. Restituisce X, funzione per il bilocale tipo, note."""
    tip = g["tipologia"].fillna("altro")
    cnt = tip.value_counts()
    liv = [t for t in cnt.index if cnt[t] >= min_liv]
    tip2 = tip.where(tip.isin(liv), "altro")
    livelli = sorted(tip2.unique())
    ref = rif if rif in livelli else cnt.index[0] if cnt.index[0] in livelli else livelli[0]
    dummy = [l for l in livelli if l != ref]
    if len(g) < 4 * (2 + len(dummy)):          # troppo pochi dati per tante dummy
        dummy, tip2 = [], tip2
    cols = [np.ones(len(g)), np.log(g["mq"].values)] + [(tip2 == l).values.astype(float) for l in dummy]
    X = np.column_stack(cols)
    x0 = np.array([1.0, math.log(mq_tipo)] + [0.0] * len(dummy))
    return X, x0, (ref if len(dummy) or len(livelli) == 1 else "n/d"), dummy


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


# ----------------------------------------------------------------------------- Moran
def moran(xy, r, perm, rng):
    n = len(r)
    D = cdist(xy, xy)
    W = 1.0 / np.maximum(D, 50.0)
    np.fill_diagonal(W, 0.0)
    S0 = W.sum()

    def stat(v):
        z = v - v.mean()
        return n / S0 * (z @ W @ z) / (z @ z)

    I = stat(r)
    sup = sum(stat(rng.permutation(r)) >= I for _ in range(perm))
    return I, (1 + sup) / (perm + 1)


# ----------------------------------------------------------------------------- variogramma
def _sph(h, n, ps, r):
    t = np.minimum(h / r, 1.0)
    return n + ps * (1.5 * t - 0.5 * t ** 3)


def _exp(h, n, ps, r):
    return n + ps * (1 - np.exp(-3 * h / r))


MODELLI = {"sferico": _sph, "esponenziale": _exp}


def gamma(modello, h, n, ps, r):
    return np.where(h > 0, MODELLI[modello](h, n, ps, r), 0.0)


def variogramma_empirico(xy, r, nb=10):
    D = cdist(xy, xy)
    iu = np.triu_indices(len(r), 1)
    h = D[iu]
    sv = 0.5 * (r[iu[0]] - r[iu[1]]) ** 2
    maxlag = h.max() / 2.0
    ed = np.linspace(0, maxlag, nb + 1)
    H, G, N = [], [], []
    for a, b in zip(ed[:-1], ed[1:]):
        m = (h > a) & (h <= b)
        if m.sum() >= 5:
            H.append(h[m].mean()); G.append(sv[m].mean()); N.append(m.sum())
    return np.array(H), np.array(G), np.array(N), maxlag


def adatta_variogramma(H, G, N, var, maxlag):
    if len(H) < 4 or var <= 0:
        return None
    best = None
    for nome, f in MODELLI.items():
        def res(p, f=f):
            return (f(H, p[0], p[1], p[2]) - G) * np.sqrt(N)
        x0 = [0.3 * var, 0.7 * var, maxlag / 2.0]
        lb = [0.0, 1e-9, 50.0]
        ub = [2.0 * var, 3.0 * var, maxlag * 2.0]
        try:
            s = least_squares(res, x0, bounds=(lb, ub))
        except Exception:
            continue
        if best is None or s.cost < best["cost"]:
            best = dict(modello=nome, nugget=float(s.x[0]), psill=float(s.x[1]), range=float(s.x[2]), cost=float(s.cost))
    return best


# ----------------------------------------------------------------------------- kriging ordinario
def krig(xy, r, vg, xy0):
    n = len(r)
    G = gamma(vg["modello"], cdist(xy, xy), vg["nugget"], vg["psill"], vg["range"])
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = G; A[:n, n] = 1.0; A[n, :n] = 1.0
    G0 = gamma(vg["modello"], cdist(xy, xy0), vg["nugget"], vg["psill"], vg["range"])
    B = np.vstack([G0, np.ones((1, xy0.shape[0]))])
    try:
        S = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        S = np.linalg.lstsq(A, B, rcond=None)[0]
    lam, mu = S[:n], S[n]
    media = lam.T @ r
    var = np.maximum((lam * G0).sum(0) + mu, 0.0)
    return media, var


# ----------------------------------------------------------------------------- analisi di una città/operazione
def metriche(prezzo, pred_eur, pred_log, y):
    return float(np.sqrt(np.mean((pred_log - y) ** 2))), float(np.mean(np.abs(pred_eur - prezzo)))


def analizza(g, op, a, org, rng):
    lat0, lon0 = org
    citta = g["citta"].iloc[0]
    g = g.copy().reset_index(drop=True)
    xy_all = proietta(g["latitudine"].values, g["longitudine"].values, lat0, lon0)
    xy_all, n_mossi = separa_duplicati(xy_all, a.jitter, rng)
    g["x"], g["y"] = xy_all[:, 0], xy_all[:, 1]
    isbil = g["tipologia"].str.lower() == "bilocale"
    nval, nbil = len(g), int(isbil.sum())
    modalita = "bilocali" if nbil >= a.soglia_bilocali else "ibrido"
    s = g[isbil].reset_index(drop=True) if modalita == "bilocali" else g
    n = len(s)
    riga = dict(operazione=op, citta=citta, n_valide=nval, n_bilocali=nbil, modalita=modalita, n_usati=n,
                posizioni_spostate=n_mossi)
    out = dict(riga=riga, punti=None, krig=None, g=g, s=s)
    if n < 10:
        riga.update(kriging_eseguito="no", motivo="campione troppo piccolo (n=%d < 10)" % n, giudizio="usare la media")
        return out
    xy = s[["x", "y"]].values
    y = np.log(s["prezzo"].values)
    X, x0, ref, dummy = disegno(s, a.mq_tipo)
    b = ols(X, y)
    r = y - X @ b
    r2 = 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()
    I, p = moran(xy, r, a.permutazioni, rng)
    riga.update(r2_regressione=round(r2, 3), moran_I=round(I, 3), moran_p=round(p, 3))
    ok_n, ok_p = n >= a.n_min, (p <= a.p_max and I > 0)
    vg = None
    if ok_n or a.forza:
        H, Gm, Np, maxlag = variogramma_empirico(xy, r)
        vg = adatta_variogramma(H, Gm, Np, float(np.var(r, ddof=1)), maxlag)
        out["vg_emp"] = (H, Gm, Np)
    esegui = (ok_n and ok_p) or (a.forza and n >= 10)
    if not esegui:
        riga["kriging_eseguito"] = "no"
        riga["motivo"] = ("n=%d < %d" % (n, a.n_min)) if not ok_n else ("nessuna struttura spaziale (Moran I=%.2f, p=%.2f)" % (I, p))
        if a.forza:
            riga["motivo"] = "campione troppo piccolo"
    elif vg is None:
        esegui = False
        riga["kriging_eseguito"] = "no"; riga["motivo"] = "variogramma non stimabile"
    # ----- cross-validation LOO (sempre: regressione e media; kriging se eseguito)
    p_med, p_reg, p_kri = np.zeros(n), np.zeros(n), np.full(n, np.nan)
    for i in range(n):
        m = np.arange(n) != i
        bi = ols(X[m], y[m])
        p_med[i] = y[m].mean()
        p_reg[i] = X[i] @ bi
        if esegui:
            ri = y[m] - X[m] @ bi
            k, _ = krig(xy[m], ri, vg, xy[i:i + 1])
            p_kri[i] = p_reg[i] + k[0]
    prezzo = s["prezzo"].values
    sb = (s["tipologia"].str.lower() == "bilocale").values
    rm = metriche(prezzo, np.exp(p_med), p_med, y)
    rr_ = metriche(prezzo, np.exp(p_reg), p_reg, y)
    riga.update(rmse_log_media=round(rm[0], 4), rmse_log_regr=round(rr_[0], 4),
                mae_eur_media=round(rm[1]), mae_eur_regr=round(rr_[1]))
    if sb.sum() >= 3:
        riga.update(mae_eur_bilocali_media=round(metriche(prezzo[sb], np.exp(p_med[sb]), p_med[sb], y[sb])[1]),
                    mae_eur_bilocali_regr=round(metriche(prezzo[sb], np.exp(p_reg[sb]), p_reg[sb], y[sb])[1]))
    pr = pd.DataFrame(dict(operazione=op, citta=citta, id_annuncio=s["id_annuncio"].values,
                           tipologia=s["tipologia"].values, mq=s["mq"].values, prezzo=prezzo,
                           latitudine=s["latitudine"].values, longitudine=s["longitudine"].values,
                           pred_regr_loo=np.exp(p_reg).round(0), residuo_regr_pct=((prezzo / np.exp(p_reg) - 1) * 100).round(1),
                           link=s["link"].values if "link" in s.columns else ""))
    out["x"], out["y_"] = xy, y
    if esegui:
        rk = metriche(prezzo, np.exp(p_kri), p_kri, y)
        riga.update(kriging_eseguito="si", motivo="", modello_variogramma=vg["modello"],
                    nugget=round(vg["nugget"], 4), psill=round(vg["psill"], 4), range_m=round(vg["range"]),
                    quota_nugget=round(vg["nugget"] / (vg["nugget"] + vg["psill"]), 3),
                    rmse_log_krig=round(rk[0], 4), mae_eur_krig=round(rk[1]))
        if sb.sum() >= 3:
            riga["mae_eur_bilocali_krig"] = round(metriche(prezzo[sb], np.exp(p_kri[sb]), p_kri[sb], y[sb])[1])
        mg_r = 1 - rk[0] / rr_[0]
        mg_m = 1 - rk[0] / rm[0]
        riga.update(miglioramento_vs_regr_pct=round(100 * mg_r, 1), miglioramento_vs_media_pct=round(100 * mg_m, 1))
        riga["giudizio"] = "kriging utile" if (mg_r >= a.min_miglioramento and mg_m > 0) else \
            ("il kriging non migliora la regressione: usare la regressione" if mg_m > 0 else "usare la media")
        pr["pred_krig_loo"] = np.exp(p_kri).round(0)
        pr["residuo_krig_pct"] = ((prezzo / np.exp(p_kri) - 1) * 100).round(1)
        out["krig"] = dict(vg=vg, b=b, X=X, x0=x0, y=y, xy=xy, r=r, p_kri=p_kri, p_reg=p_reg, ref=ref, dummy=dummy)
    else:
        riga["giudizio"] = "usare la media"
    out["punti"] = pr
    return out


# ----------------------------------------------------------------------------- griglia
def predici_griglia(ris, xs, ys, mask, a):
    K = ris["krig"]
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel()][mask.ravel()]
    m, v = krig(K["xy"], K["r"], K["vg"], pts)
    base = K["x0"] @ K["b"]
    ml = base + m
    sd = np.sqrt(v)
    est = np.full(XX.size, np.nan); inc = est.copy(); lo = est.copy(); hi = est.copy()
    idx = np.where(mask.ravel())[0]
    est[idx] = np.exp(ml); inc[idx] = (np.exp(sd) - 1) * 100
    lo[idx] = np.exp(ml - Z80 * sd); hi[idx] = np.exp(ml + Z80 * sd)
    sh = XX.shape
    return est.reshape(sh), inc.reshape(sh), lo.reshape(sh), hi.reshape(sh)


def maschera(xs, ys, pts, raggio):
    XX, YY = np.meshgrid(xs, ys)
    d, _ = cKDTree(pts).query(np.c_[XX.ravel(), YY.ravel()])
    return (d <= raggio).reshape(XX.shape)


# ----------------------------------------------------------------------------- grafica
def rgba(val, cmap, vmin, vmax, alpha=0.72):
    nrm = mcolors.Normalize(vmin, vmax, clip=True)
    img = plt.get_cmap(cmap)(nrm(np.nan_to_num(val, nan=vmin)))
    img[..., 3] = np.where(np.isnan(val), 0.0, alpha)
    return img


def png_b64(img):
    buf = io.BytesIO()
    plt.imsave(buf, img, format="png", origin="lower")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def limiti(val, lo=5, hi=95):
    v = val[~np.isnan(val)]
    a, b = np.percentile(v, [lo, hi])
    if b - a < 1e-9:
        b = a + 1.0
    return float(a), float(b)


def estensione(xs, ys, c, lat0, lon0):
    la0, lo0 = xy_a_ll(xs[0] - c / 2, ys[0] - c / 2, lat0, lon0)
    la1, lo1 = xy_a_ll(xs[-1] + c / 2, ys[-1] + c / 2, lat0, lon0)
    return float(la0), float(lo0), float(la1), float(lo1)


def sfondo(ax):
    """Sfondo cartografico (richiede contextily e internet). Prova Esri grigio chiaro, poi OpenStreetMap."""
    if cx is None:
        return False
    for nome in ("Esri.WorldGrayCanvas", "OpenStreetMap.Mapnik"):
        try:
            prov = cx.providers
            for pezzo in nome.split("."):
                prov = prov[pezzo]
            cx.add_basemap(ax, source=prov, crs="EPSG:3857", attribution_size=6)
            return True
        except Exception:
            continue
    return False


def hex_cmap(cmap, n=6):
    c = plt.get_cmap(cmap)
    return [mcolors.to_hex(c(i / (n - 1))) for i in range(n)]


def png_mappa(path, titolo, panels, pts_ll, pts_val, ext, lat0, lon0):
    """panels: lista di (titolo, array, cmap, vmin, vmax, etichetta_colorbar)."""
    la0, lo0, la1, lo1 = ext
    x0, y0 = merc(lo0, la0); x1, y1 = merc(lo1, la1)
    px, py = merc(pts_ll[:, 1], pts_ll[:, 0])
    rap = min(max((y1 - y0) / max(x1 - x0, 1.0), 0.55), 1.3)
    fig, axs = plt.subplots(1, len(panels), figsize=(6.4 * len(panels) + 0.8, 6.4 * rap + 1.1))
    axs = np.atleast_1d(axs)
    usato = False
    for ax, (tt, arr, cmap, vmin, vmax, lab) in zip(axs, panels):
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        usato = sfondo(ax) or usato
        ax.imshow(rgba(arr, cmap, vmin, vmax), extent=(x0, x1, y0, y1), origin="lower", zorder=3, interpolation="bilinear")
        if pts_val is not None and "inc" not in tt.lower():
            sc = ax.scatter(px, py, c=pts_val, cmap=cmap, vmin=vmin, vmax=vmax, s=34, edgecolor="white", linewidth=0.8, zorder=5)
        else:
            ax.scatter(px, py, s=14, c="#222222", edgecolor="white", linewidth=0.5, zorder=5)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(tt, fontsize=11, loc="left")
        sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, shrink=0.8)
        cb.set_label(lab, fontsize=9)
    fig.suptitle(titolo, fontsize=13, fontweight="bold", x=0.02, ha="left")
    if not usato:
        fig.text(0.02, 0.01, "Sfondo cartografico non disponibile (installa contextily e usa internet): mappa senza basemap.", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def png_diagnostica(path, titolo, ris):
    K = ris["krig"]; H, G, N = ris["vg_emp"]; vg = K["vg"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    hh = np.linspace(0.01, max(H.max() * 1.1, vg["range"]), 200)
    a1.plot(H, G, "o", color="#1f77b4", label="empirico")
    a1.plot(hh, MODELLI[vg["modello"]](hh, vg["nugget"], vg["psill"], vg["range"]), "-", color="#d62728", label=vg["modello"])
    a1.axhline(vg["nugget"] + vg["psill"], ls=":", color="grey")
    a1.axvline(vg["range"], ls=":", color="grey")
    a1.set_xlabel("distanza (m)"); a1.set_ylabel("semivarianza dei residui (log)")
    a1.set_title("Variogramma  (nugget %.3f, psill %.3f, range %d m)" % (vg["nugget"], vg["psill"], vg["range"]), fontsize=10)
    a1.legend(frameon=False)
    obs = np.exp(K["y"]); k = np.exp(K["p_kri"]); rg = np.exp(K["p_reg"])
    a2.scatter(obs, rg, s=18, color="#999999", label="solo regressione")
    a2.scatter(obs, k, s=22, color="#d62728", label="regressione + kriging")
    lim = [min(obs.min(), k.min()) * 0.9, max(obs.max(), k.max()) * 1.1]
    a2.plot(lim, lim, "k--", lw=0.8)
    a2.set_xlim(lim); a2.set_ylim(lim)
    a2.set_xlabel("osservato"); a2.set_ylabel("previsto (leave-one-out)")
    a2.set_title("Cross-validation", fontsize=10); a2.legend(frameon=False)
    fig.suptitle(titolo, fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------- HTML
HTML = r"""<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mappe kriging - GDC Immobiliare</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
html,body{height:100%;margin:0;font-family:system-ui,Segoe UI,Arial,sans-serif;color:#1d2433}
#map{position:absolute;inset:0}
#pannello{position:absolute;z-index:1000;top:12px;left:56px;background:rgba(255,255,255,.96);border-radius:10px;padding:12px 14px;box-shadow:0 2px 14px rgba(0,0,0,.25);width:285px;font-size:13px;max-height:calc(100% - 40px);overflow:auto}
#pannello h1{font-size:15px;margin:0 0 8px}
#pannello label{display:block;margin:4px 0;cursor:pointer}
#pannello label.off{color:#9aa1b0;cursor:not-allowed}
#pannello select{width:100%;padding:5px;margin:2px 0 6px;font-size:13px}
.sez{margin-top:8px;padding-top:6px;border-top:1px solid #e3e6ee}
#info{font-size:12px;color:#444;margin-top:8px;line-height:1.4}
#legenda{position:absolute;z-index:1000;bottom:26px;left:12px;background:rgba(255,255,255,.96);border-radius:10px;padding:8px 12px;box-shadow:0 2px 14px rgba(0,0,0,.25);font-size:12px;min-width:240px;max-width:300px}
.leg{margin:4px 0 8px}.barra{height:12px;border-radius:3px;margin:4px 0}
.estremi{display:flex;justify-content:space-between}
.badge{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;background:#eef1f7;margin:1px 0}
.ok{background:#dff3e4}.ind{background:#fdf0cf}.no{background:#fbe3e0}
@media(max-width:600px){#pannello{width:225px;left:50px}}
</style></head><body>
<div id="map"></div>
<div id="pannello">
 <h1>Kriging prezzi e canoni</h1>
 <select id="citta"></select>
 <div class="sez"><b>Livelli stimati</b> (si possono sovrapporre)
  <label id="r_l_v"><input type="checkbox" id="l_v"> Prezzo di vendita stimato</label>
  <label id="r_l_a"><input type="checkbox" id="l_a"> Canone di affitto stimato</label>
  <label id="r_l_r"><input type="checkbox" id="l_r"> Rendimento (canone×12 / prezzo)</label>
  <label id="r_l_iv"><input type="checkbox" id="l_iv"> Incertezza prezzo di vendita</label>
  <label id="r_l_ia"><input type="checkbox" id="l_ia"> Incertezza canone</label>
 </div>
 <div class="sez"><b>Annunci</b> (blu = più economico del previsto, rosso = più caro)
  <label><input type="checkbox" id="p_v" checked> Vendite (bordo bianco)</label>
  <label><input type="checkbox" id="p_a"> Affitti (bordo nero)</label>
 </div>
 <div class="sez">Trasparenza livelli <input type="range" id="trasp" min="0.15" max="1" step="0.05" value="1" style="width:100%"></div>
 <div id="info"></div>
</div>
<div id="legenda"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
var DATA=/*DATA*/null;
var base={
 "Grigio chiaro (Esri)":L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",{maxNativeZoom:16,maxZoom:19,attribution:"Esri, HERE, Garmin"}),
 "Strade (Esri)":L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",{maxZoom:19,attribution:"Esri"}),
 "Satellite (Esri)":L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",{maxZoom:19,attribution:"Esri"})
};
var map=L.map("map",{zoomControl:true}).setView([44.5,10.5],6);
base["Grigio chiaro (Esri)"].addTo(map);L.control.layers(base,null,{position:"topright"}).addTo(map);
var lyr={};
function esc(s){var d=document.createElement("div");d.textContent=s==null?"":String(s);return d.innerHTML}
function fmt(v,u){if(v==null)return "-";return u==="%"?v.toFixed(1):Math.round(v).toLocaleString("it-IT")}
function specs(C){
 var o={},V=C.vendita||{},A=C.affitto||{},R=C.rendimento||null;
 if(V.stima)o.l_v={img:V.stima,b:V.bounds,vmin:V.vmin,vmax:V.vmax,stops:V.stops,u:"€",tit:V.titolo,aff:V.affid};
 if(A.stima)o.l_a={img:A.stima,b:A.bounds,vmin:A.vmin,vmax:A.vmax,stops:A.stops,u:"€",tit:A.titolo,aff:A.affid};
 if(R)o.l_r={img:R.stima,b:R.bounds,vmin:R.vmin,vmax:R.vmax,stops:R.stops,u:"%",tit:R.titolo,aff:R.affid};
 if(V.inc)o.l_iv={img:V.inc,b:V.bounds,vmin:V.inc_min,vmax:V.inc_max,stops:V.inc_stops,u:"%",tit:"Incertezza prezzo di vendita (± %, intervallo 80%)",aff:V.affid};
 if(A.inc)o.l_ia={img:A.inc,b:A.bounds,vmin:A.inc_min,vmax:A.inc_max,stops:A.inc_stops,u:"%",tit:"Incertezza canone (± %, intervallo 80%)",aff:A.affid};
 return o;
}
function chk(id){return document.getElementById(id).checked}
function set(id,v){document.getElementById(id).checked=v}
function difetti(C){
 set("l_v",false);set("l_a",false);set("l_r",false);set("l_iv",false);set("l_ia",false);
 var S=specs(C);
 if(S.l_r)set("l_r",true);else if(S.l_v)set("l_v",true);else if(S.l_a)set("l_a",true);
 set("p_v",!!(C.punti.vendita&&C.punti.vendita.length));set("p_a",false);
}
function aggiorna(nuova){
 var C=DATA[document.getElementById("citta").value];if(!C)return;
 if(nuova)difetti(C);
 for(var k in lyr){if(lyr[k]){map.removeLayer(lyr[k]);lyr[k]=null}}
 var S=specs(C),ordine=["l_iv","l_ia","l_v","l_a","l_r"],leg="",op=parseFloat(document.getElementById("trasp").value),z=10;
 ordine.forEach(function(id){
  var r=document.getElementById("r_"+id);
  r.className=S[id]?"":"off";document.getElementById(id).disabled=!S[id];
  if(!S[id]){document.getElementById(id).checked=false;return}
  if(chk(id)){var s=S[id];
   lyr[id]=L.imageOverlay(s.img,[[s.b[0],s.b[1]],[s.b[2],s.b[3]]],{opacity:op,interactive:false,zIndex:z++}).addTo(map);
   leg="<div class='leg'><b>"+esc(s.tit)+"</b><div class='barra' style='background:linear-gradient(90deg,"+s.stops.join(",")+")'></div><div class='estremi'><span>"+fmt(s.vmin,s.u)+(s.u==="%"?" %":" €")+"</span><span>"+fmt(s.vmax,s.u)+(s.u==="%"?" %":" €")+"</span></div></div>"+leg}
 });
 ["vendita","affitto"].forEach(function(o){
  var id=o==="vendita"?"p_v":"p_a";
  if(chk(id)){var g=L.layerGroup();
   (C.punti[o]||[]).forEach(function(p){
    var m=L.circleMarker([p.lat,p.lon],{radius:6,color:o==="vendita"?"#fff":"#222",weight:1.8,fillColor:p.col,fillOpacity:.95});
    var h="<b>"+esc(p.tip)+"</b> – "+esc(p.mq)+" m²<br>"+(o==="vendita"?"Prezzo: ":"Canone: ")+fmt(p.prezzo,"€")+" €"+(o==="affitto"?"/mese":"");
    if(p.atteso!=null)h+="<br>Atteso dal modello: "+fmt(p.atteso,"€")+" € ("+(p.res>0?"+":"")+p.res.toFixed(0)+"%)";
    if(/^https?:\/\//.test(p.link||""))h+="<br><a target='_blank' rel='noopener' href='"+esc(p.link)+"'>apri annuncio</a>";
    m.bindPopup(h);g.addLayer(m)});
   lyr[id]=g.addTo(map)}
 });
 var info="<b>"+esc(C.nome)+"</b><br>";
 ["vendita","affitto"].forEach(function(o){
  var D=C[o];if(!D){info+="<span class='badge no'>"+o+": nessun annuncio</span><br>";return}
  if(D.stima)info+="<span class='badge "+(D.affid==="utile"?"ok":"ind")+"'>"+o+": kriging "+(D.affid==="utile"?"utile":"eseguito, ma non migliora i modelli semplici (indicativo)")+"</span> n="+D.n+"<br>";
  else info+="<span class='badge no'>"+o+": nessun kriging</span> n="+D.n+" – "+esc(D.motivo||"")+"<br>";
 });
 if(C.rendimento)info+="<span class='badge "+(C.rendimento.affid==="affidabile"?"ok":"ind")+"'>rendimento "+(C.rendimento.affid==="affidabile"?"affidabile":"indicativo")+"</span><br>";
 else info+="<span class='badge no'>rendimento non calcolabile</span> (serve il kriging eseguito sia per vendite sia per affitti)<br>";
 document.getElementById("info").innerHTML=info;
 document.getElementById("legenda").innerHTML=leg||"<b>Nessun livello stimato attivo</b><br>Seleziona un livello nel pannello (se disponibile).";
 if(nuova)map.fitBounds(C.vista,{padding:[30,30]});
}
var sel=document.getElementById("citta");
Object.keys(DATA).sort().forEach(function(k){var o=document.createElement("option");o.value=k;o.textContent=DATA[k].nome;sel.appendChild(o)});
sel.addEventListener("change",function(){aggiorna(true)});
["l_v","l_a","l_r","l_iv","l_ia","p_v","p_a"].forEach(function(i){document.getElementById(i).addEventListener("change",function(){aggiorna(false)})});
document.getElementById("trasp").addEventListener("input",function(){var v=parseFloat(this.value);for(var k in lyr){if(lyr[k]&&lyr[k].setOpacity)lyr[k].setOpacity(v)}});
aggiorna(true);
</script></body></html>"""


# ----------------------------------------------------------------------------- main
def main():
    base = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Kriging di prezzi e canoni per città (solo file *_GEO.csv).")
    ap.add_argument("--input-dir", default=str(base / "analisi_modelli"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--vendite", default="output_vendite_GEO.csv")
    ap.add_argument("--affitti", default="output_affitti_GEO.csv")
    ap.add_argument("--n-min", type=int, default=20, help="minimo di annunci per fare il kriging (default 20)")
    ap.add_argument("--p-max", type=float, default=0.15, help="massimo p-value del test di Moran (default 0.15)")
    ap.add_argument("--mq-tipo", type=float, default=50.0, help="metratura del bilocale tipo (default 50)")
    ap.add_argument("--soglia-bilocali", type=int, default=25, help="se la città ha >= N bilocali usa solo quelli (default 25)")
    ap.add_argument("--min-miglioramento", type=float, default=0.03, help="miglioramento minimo dell'RMSE vs sola regressione (default 0.03)")
    ap.add_argument("--cella", type=float, default=100.0, help="lato cella della griglia in metri (default 100)")
    ap.add_argument("--raggio", type=float, default=700.0, help="la stima si mostra solo entro N metri da un annuncio (default 700)")
    ap.add_argument("--jitter", type=float, default=15.0, help="spostamento (m) per posizioni identiche (default 15)")
    ap.add_argument("--permutazioni", type=int, default=999)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--solo", default="", help="lista di città separate da virgola (default: tutte)")
    ap.add_argument("--forza", action="store_true", help="fa il kriging anche dove il test di Moran non e' significativo (n>=10)")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--no-html", action="store_true")
    a = ap.parse_args()

    ind = Path(a.input_dir)
    out = Path(a.out_dir) if a.out_dir else ind / "out_kriging"
    (out / "png").mkdir(parents=True, exist_ok=True)
    (out / "griglie").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 78)
    print("KRIGING PER CITTA'   (input: %s)" % ind)
    print("=" * 78)
    dati = {}
    for op, f in (("vendita", a.vendite), ("affitto", a.affitti)):
        p = ind / f
        if not p.exists():
            sys.exit("ERRORE: file non trovato: %s" % p)
        dati[op] = prepara(leggi(p), f)
    sel = [c.strip().lower().replace(" ", "") for c in a.solo.split(",") if c.strip()]
    citta_tutte = sorted(set(dati["vendita"].citta) | set(dati["affitto"].citta))
    if sel:
        citta_tutte = [c for c in citta_tutte if c in sel]
    print("Città: %d   |   contextily %s   |   n-min %d, p-max %.2f, bilocale tipo %.0f m², soglia bilocali %d"
          % (len(citta_tutte), "disponibile" if cx else "NON installato (PNG senza sfondo)", a.n_min, a.p_max, a.mq_tipo, a.soglia_bilocali))
    rng = np.random.default_rng(a.seed)

    righe, punti_tutti, rend_righe, html = [], [], [], {}
    for c in citta_tutte:
        gs = {op: dati[op][dati[op].citta == c] for op in OPS}
        tutti = pd.concat([gs[o] for o in OPS if len(gs[o])])
        org = (float(tutti.latitudine.mean()), float(tutti.longitudine.mean()))
        res = {}
        for op in OPS:
            if len(gs[op]):
                res[op] = analizza(gs[op], op, a, org, rng)
                righe.append(res[op]["riga"])
                if res[op]["punti"] is not None:
                    punti_tutti.append(res[op]["punti"])
        # ---- griglia comune
        esec = {op: res[op]["krig"] is not None for op in res}
        utili = {op: (esec[op] and res[op]["riga"].get("giudizio") == "kriging utile") for op in res}
        entry = dict(nome=c.capitalize(), punti={}, vista=None)
        allxy = np.vstack([res[o]["g"][["x", "y"]].values for o in res])
        la_a, lo_a = xy_a_ll(allxy[:, 0].min() - 300, allxy[:, 1].min() - 300, *org)
        la_b, lo_b = xy_a_ll(allxy[:, 0].max() + 300, allxy[:, 1].max() + 300, *org)
        entry["vista"] = [[float(la_a), float(lo_a)], [float(la_b), float(lo_b)]]
        cella = a.cella
        xmin, xmax = allxy[:, 0].min() - a.raggio, allxy[:, 0].max() + a.raggio
        ymin, ymax = allxy[:, 1].min() - a.raggio, allxy[:, 1].max() + a.raggio
        while ((xmax - xmin) / cella) * ((ymax - ymin) / cella) > 40000:
            cella *= 1.5
        xs = np.arange(xmin, xmax + cella, cella); ys = np.arange(ymin, ymax + cella, cella)
        ext = estensione(xs, ys, cella, *org)
        gr = {}
        for op in res:
            r_ = res[op]
            pr = r_["punti"]
            pts = []
            if pr is not None:
                col = "residuo_krig_pct" if "residuo_krig_pct" in pr.columns else "residuo_regr_pct"
                att = "pred_krig_loo" if "pred_krig_loo" in pr.columns else "pred_regr_loo"
                nr = mcolors.Normalize(-40, 40, clip=True)
                for _, q in pr.iterrows():
                    pts.append(dict(lat=q.latitudine, lon=q.longitudine, tip=q.tipologia, mq=float(q.mq), prezzo=float(q.prezzo),
                                    atteso=float(q[att]), res=float(q[col]), link=q.link,
                                    col=mcolors.to_hex(plt.get_cmap("RdBu_r")(nr(q[col])))))
            entry["punti"][op] = pts
            d_op = dict(n=int(r_["riga"]["n_usati"]), modalita=r_["riga"]["modalita"], motivo=r_["riga"].get("motivo", ""))
            if esec[op]:
                mask = maschera(xs, ys, r_["s"][["x", "y"]].values, a.raggio)
                est, inc, lo, hi = predici_griglia(r_, xs, ys, mask, a)
                gr[op] = dict(est=est, inc=inc, lo=lo, hi=hi, mask=mask)
                vmin, vmax = limiti(est); imin, imax = limiti(inc)
                cm_s, cm_i = "YlOrRd", "PuBu"
                d_op.update(affid=("utile" if utili[op] else "indicativo"), stima=png_b64(rgba(est, cm_s, vmin, vmax)), inc=png_b64(rgba(inc, cm_i, imin, imax)),
                            bounds=ext, vmin=vmin, vmax=vmax, stops=hex_cmap(cm_s), inc_min=imin, inc_max=imax, inc_stops=hex_cmap(cm_i),
                            titolo=("Prezzo di vendita stimato" if op == "vendita" else "Canone mensile stimato") + " – bilocale %.0f m²" % a.mq_tipo,
                            nota="mappa valida entro %d m dagli annunci" % a.raggio)
                XX, YY = np.meshgrid(xs, ys)
                la, lo_ = xy_a_ll(XX.ravel(), YY.ravel(), *org)
                m_ = mask.ravel()
                pd.DataFrame(dict(latitudine=la[m_].round(6), longitudine=lo_[m_].round(6), stima=est.ravel()[m_].round(0),
                                  incertezza_pct=inc.ravel()[m_].round(1), stima_lo80=lo.ravel()[m_].round(0), stima_hi80=hi.ravel()[m_].round(0))
                             ).to_csv(out / "griglie" / ("griglia_%s_%s.csv" % (op, c)), sep=";", decimal=",", index=False, encoding="utf-8-sig")
                if not a.no_png:
                    ll = r_["punti"][["latitudine", "longitudine"]].values
                    tt = "%s – %s · bilocale tipo %.0f m² · %s, n=%d" % (c.capitalize(), op, a.mq_tipo, r_["riga"]["modalita"], r_["riga"]["n_usati"])
                    png_mappa(out / "png" / ("mappa_%s_%s.png" % (op, c)), tt,
                              [("Valore stimato", est, cm_s, vmin, vmax, UNITA[op]), ("Incertezza (± %, intervallo 80%)", inc, cm_i, imin, imax, "± %")],
                              ll, r_["punti"]["prezzo"].values, ext, *org)
                    png_diagnostica(out / "png" / ("diagnostica_%s_%s.png" % (op, c)), "%s – %s: diagnostica" % (c.capitalize(), op), r_)
            entry[op] = d_op
        # ---- rendimento
        if all(esec.get(o) for o in OPS):
            mk = gr["vendita"]["mask"] & gr["affitto"]["mask"]
            rend = np.where(mk, gr["affitto"]["est"] * 12.0 / gr["vendita"]["est"] * 100.0, np.nan)
            if np.isfinite(rend).any():
                vmin, vmax = limiti(rend); cm_r = "RdYlGn"
                entry["rendimento"] = dict(stima=png_b64(rgba(rend, cm_r, vmin, vmax)), bounds=ext, vmin=vmin, vmax=vmax, stops=hex_cmap(cm_r),
                                           titolo="Rendimento stimato (canone×12 / prezzo, %)", n=int(np.isfinite(rend).sum()), modalita="", affid=("affidabile" if all(utili.get(o) for o in OPS) else "indicativo"),
                                           nota="calcolato dove entrambe le stime sono affidabili")
                XX, YY = np.meshgrid(xs, ys)
                la, lo_ = xy_a_ll(XX.ravel(), YY.ravel(), *org); m_ = np.isfinite(rend.ravel())
                pd.DataFrame(dict(latitudine=la[m_].round(6), longitudine=lo_[m_].round(6), prezzo_stimato=gr["vendita"]["est"].ravel()[m_].round(0),
                                  canone_stimato=gr["affitto"]["est"].ravel()[m_].round(0), rendimento_pct=rend.ravel()[m_].round(2))
                             ).to_csv(out / "griglie" / ("rendimento_%s.csv" % c), sep=";", decimal=",", index=False, encoding="utf-8-sig")
                v = rend[np.isfinite(rend)]
                rend_righe.append(dict(citta=c, celle=int(v.size), rend_min=round(v.min(), 2), rend_p25=round(np.percentile(v, 25), 2),
                                       rend_mediano=round(np.median(v), 2), rend_p75=round(np.percentile(v, 75), 2), rend_max=round(v.max(), 2),
                                       mq_tipo=a.mq_tipo, affidabilita=("affidabile" if all(utili.get(o) for o in OPS) else "indicativo")))
                if not a.no_png:
                    ll = np.vstack([res[o]["punti"][["latitudine", "longitudine"]].values for o in OPS])
                    png_mappa(out / "png" / ("mappa_rendimento_%s.png" % c), "%s – rendimento lordo stimato (canone×12 / prezzo), bilocale tipo %.0f m²" % (c.capitalize(), a.mq_tipo),
                              [("Rendimento stimato (%)", rend, cm_r, vmin, vmax, "% annuo")], ll, None, ext, *org)
        html[c] = entry
        # ---- riga a terminale
        for op in res:
            q = res[op]["riga"]
            if q.get("kriging_eseguito") == "si":
                print("  %-13s %-8s n=%3d (%s)  Moran I=%5.2f p=%.3f | range %5d m, quota nugget %.2f | RMSE vs regr %+5.1f%%, vs media %+5.1f%% -> %s"
                      % (c, op, q["n_usati"], q["modalita"], q["moran_I"], q["moran_p"], q["range_m"], q["quota_nugget"],
                         q["miglioramento_vs_regr_pct"], q["miglioramento_vs_media_pct"], q["giudizio"]))
            else:
                print("  %-13s %-8s n=%3d (%s)  %s  -> %s"
                      % (c, op, q["n_usati"], q["modalita"], ("Moran I=%.2f p=%.3f; " % (q["moran_I"], q["moran_p"]) if "moran_p" in q else "") + q.get("motivo", ""), q["giudizio"]))

    # ---- scrittura CSV
    wr = dict(sep=";", decimal=",", index=False, encoding="utf-8-sig")
    pd.DataFrame(righe).to_csv(out / "kriging_riepilogo.csv", **wr)
    if punti_tutti:
        pd.concat(punti_tutti).to_csv(out / "kriging_punti.csv", **wr)
    if rend_righe:
        pd.DataFrame(rend_righe).to_csv(out / "kriging_rendimento_riepilogo.csv", **wr)
    if not a.no_html:
        s = json.dumps(html, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        (out / "mappe_kriging.html").write_text(HTML.replace("/*DATA*/null", s), encoding="utf-8")
    nk = sum(1 for r in righe if r.get("kriging_eseguito") == "si")
    nu = sum(1 for r in righe if r.get("giudizio") == "kriging utile")
    print("-" * 78)
    print("Analisi: %d (città x operazione) | kriging eseguito: %d | kriging utile: %d | rendimento mappato: %d città" % (len(righe), nk, nu, len(rend_righe)))
    print("Output in: %s   (%.0f s)" % (out, time.time() - t0))
    if not a.no_html:
        print("Mappa interattiva: %s" % (out / "mappe_kriging.html"))


if __name__ == "__main__":
    main()
