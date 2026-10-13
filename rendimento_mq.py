# -*- coding: utf-8 -*-
"""
GDC Immobiliare - rendimento del BILOCALE a parita' di dimensione: prezzi e canoni al metro quadro, per citta'.

Input : analisi_modelli/output_vendite.csv, output_affitti.csv  (considerati GIA' PULITI, stessa preparazione di analisi_prezzi.py)
Output: analisi_modelli/out_rendimento_mq/
          rendimento_mq_citta.csv, 00_riepilogo_rendimento_mq.txt, g_rendimento_mq.png, g_confronto_con_senza_mq.png

Per ogni annuncio (solo la tipologia scelta, default Bilocale, e solo annunci con metri quadri validi):
  prezzo_mq = prezzo di vendita / mq      canone_mq = canone mensile / mq      spese_mq = spese condominiali mensili / mq
Per ogni citta' (medie aritmetiche degli annunci; spese_mq calcolate sull'insieme di vendite e affitti con spese e mq):
  rendimento_lordo_mq_%  = (canone_mq_medio + spese_mq_medio) * 12 / prezzo_mq_medio * 100   (spese pagate dall'inquilino, incluse)
  rendimento_netto_mq_%  = canone_mq_medio * 12 / prezzo_mq_medio * 100                       (solo canone)
  rendimento_netto_senza_mq_% = canone medio * 12 / prezzo medio * 100   (metodo precedente, per vedere l'effetto del controllo dei mq)
  mq_medio_vendite, mq_medio_affitti, rapporto_mq  -> se i bilocali in vendita e in affitto hanno dimensioni molto diverse (rapporto fuori
  da 0,85-1,15) la colonna comparabilita' segnala "mq diversi": il confronto senza mq sarebbe distorto.
affidabilita' = "alta" se la citta' ha almeno 3 vendite e 3 affitti con mq; altrimenti "bassa". Nessuna soglia di esclusione.

NOTA SULLA MULTICOLLINEARITA'
  prezzo_mq = prezzo / mq: in un modello in scala logaritmica log(prezzo_mq) = log(prezzo) - log(mq), cioe' le tre variabili sono
  PERFETTAMENTE collineari e non possono stare insieme nello stesso modello. In scala lineare la collinearita' e' alta ma non perfetta
  (vedi il VIF stampato). Qui non si stima nessun modello: si calcolano rapporti, per cui il problema non si pone.

Uso:   python rendimento_mq.py
       python rendimento_mq.py --tipologia Trilocale
       python rendimento_mq.py --mq-min 25 --mq-max 90     (facoltativo: limita l'intervallo di metri quadri; default nessun limite)
Dipendenze: pandas, numpy, matplotlib. Richiede analisi_prezzi.py nella stessa cartella. Nessun accesso a internet.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analisi_prezzi as ap

AQUI = os.path.dirname(os.path.abspath(__file__))
BLU, ARANCIO, GRIGIO, FONDO = "#2a78d6", "#e07b2e", "#8a8f98", "#fcfcfb"
CREATI, SALTATI = [], []


def stile(ax, g="x"):
    ax.set_facecolor(FONDO); ax.grid(axis=g, color="#e6e6e3", linewidth=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def salva(fig, path):
    fig.patch.set_facecolor(FONDO); fig.tight_layout()
    try:
        fig.savefig(path, dpi=140, facecolor=FONDO); CREATI.append(os.path.basename(path))
    except OSError as e:
        SALTATI.append("%s (%s) - file probabilmente aperto in un visualizzatore" % (os.path.basename(path), e))
    plt.close(fig)


def vif_testo(d, nome):
    """VIF di (prezzo, mq, prezzo_mq) in scala lineare, sul campione degli annunci in vendita."""
    X = d[["prezzo", "mq", "ppm"]].dropna()
    if len(X) < 10:
        return "VIF %s: campione troppo piccolo" % nome
    C = np.corrcoef(X.to_numpy(float), rowvar=False)
    try:
        v = np.diag(np.linalg.inv(C))
        return "VIF %s (scala lineare): prezzo %.1f | mq %.1f | prezzo_mq %.1f   (VIF > 10 = collinearita' forte)" % (nome, v[0], v[1], v[2])
    except np.linalg.LinAlgError:
        return "VIF %s: matrice singolare (collinearita' perfetta)" % nome


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cartella", default="analisi_modelli")
    p.add_argument("--tipologia", default="Bilocale")
    p.add_argument("--mq-min", type=float, default=None)
    p.add_argument("--mq-max", type=float, default=None)
    a = p.parse_args()
    base = a.cartella if os.path.isabs(a.cartella) else os.path.join(AQUI, a.cartella)
    fv, fa = os.path.join(base, "output_vendite.csv"), os.path.join(base, "output_affitti.csv")
    for f in (fv, fa):
        if not os.path.exists(f):
            print("File non trovato:", f); return 1
    out = os.path.join(base, "out_rendimento_mq"); os.makedirs(out, exist_ok=True)

    V = ap.prepara(fv, "vendita", None, []); A = ap.prepara(fa, "affitto", None, [])
    R = ["RENDIMENTO AL METRO QUADRO - tipologia: %s" % a.tipologia, ""]
    dati = {}
    for nome, d in (("vendite", V), ("affitti", A)):
        t = d[d["tip"] == a.tipologia].copy()
        n_t = len(t)
        t = t[t["mq"].notna() & (t["mq"] > 0)]
        n_mq = len(t)
        if a.mq_min is not None:
            t = t[t["mq"] >= a.mq_min]
        if a.mq_max is not None:
            t = t[t["mq"] <= a.mq_max]
        R.append("%s: annunci di tipologia %s: %d | con mq validi: %d (%d senza mq, esclusi)%s | usati: %d" % (
            nome, a.tipologia, n_t, n_mq, n_t - n_mq,
            "" if (a.mq_min is None and a.mq_max is None) else " | filtro mq %s-%s" % (a.mq_min, a.mq_max), len(t)))
        t["x_mq"] = t["prezzo"] / t["mq"]
        t["spese_mq"] = t["spese_mese"] / t["mq"]
        dati[nome] = t
    Vt, At = dati["vendite"], dati["affitti"]
    if Vt.empty or At.empty:
        R.append("Dati insufficienti."); print("\n".join(R)); return 1
    Vt = Vt.rename(columns={"x_mq": "ppm"}); At = At.rename(columns={"x_mq": "cpm"})   # nomi nuovi: il file ha gia' una colonna prezzo_mq

    righe = []
    for c in sorted(set(Vt["citta"]) | set(At["citta"])):
        v, af = Vt[Vt.citta == c], At[At.citta == c]
        sp = pd.concat([v["spese_mq"], af["spese_mq"]]).dropna()
        r = {"citta": c, "n_vendite": len(v), "n_affitti": len(af),
             "mq_medio_vendite": v["mq"].mean(), "mq_medio_affitti": af["mq"].mean(),
             "prezzo_mq_medio": v["ppm"].mean(), "prezzo_mq_mediano": v["ppm"].median(),
             "canone_mq_medio": af["cpm"].mean(), "canone_mq_mediano": af["cpm"].median(),
             "n_spese_mq": len(sp), "spese_mq_medie": sp.mean() if len(sp) else np.nan,
             "prezzo_medio": v["prezzo"].mean(), "canone_medio": af["prezzo"].mean()}
        righe.append(r)
    K = pd.DataFrame(righe)
    K = K[(K.n_vendite >= 1) & (K.n_affitti >= 1)].copy()
    K["rendimento_lordo_mq_%"] = (K["canone_mq_medio"] + K["spese_mq_medie"]) * 12 / K["prezzo_mq_medio"] * 100
    K["rendimento_netto_mq_%"] = K["canone_mq_medio"] * 12 / K["prezzo_mq_medio"] * 100
    K["rendimento_netto_mq_da_mediane_%"] = K["canone_mq_mediano"] * 12 / K["prezzo_mq_mediano"] * 100
    K["rendimento_netto_senza_mq_%"] = K["canone_medio"] * 12 / K["prezzo_medio"] * 100
    K["rapporto_mq"] = K["mq_medio_vendite"] / K["mq_medio_affitti"]
    K["comparabilita"] = np.where(K["rapporto_mq"].between(0.85, 1.15), "mq simili", "mq diversi")
    K["affidabilita"] = np.where((K.n_vendite >= 3) & (K.n_affitti >= 3), "alta", "bassa")
    K = K.sort_values("rendimento_netto_mq_%", ascending=False, na_position="last")
    col = ["citta", "n_vendite", "n_affitti", "mq_medio_vendite", "mq_medio_affitti", "rapporto_mq", "comparabilita", "prezzo_mq_medio",
           "canone_mq_medio", "n_spese_mq", "spese_mq_medie", "rendimento_lordo_mq_%", "rendimento_netto_mq_%",
           "rendimento_netto_mq_da_mediane_%", "rendimento_netto_senza_mq_%", "affidabilita"]
    K = K[col].round(2)
    ap.scrivi_csv(K, os.path.join(out, "rendimento_mq_citta.csv"))

    R += ["", "=" * 100, "PER CITTA' - ordinate per rendimento netto al metro quadro (tipologia %s)" % a.tipologia, "=" * 100,
          K[["citta", "n_vendite", "n_affitti", "mq_medio_vendite", "mq_medio_affitti", "prezzo_mq_medio", "canone_mq_medio", "spese_mq_medie",
             "rendimento_lordo_mq_%", "rendimento_netto_mq_%", "rendimento_netto_senza_mq_%", "comparabilita", "affidabilita"]].to_string(index=False), "",
          "Lettura:",
          "- rendimento_*_mq: calcolato su prezzi e canoni al metro quadro (a parita' di dimensione). Lordo = canone + spese; netto = solo canone.",
          "- rendimento_netto_senza_mq: metodo precedente (canone medio * 12 / prezzo medio). La differenza tra le due colonne misura l'effetto dei metri quadri.",
          "- comparabilita = 'mq diversi': i bilocali in vendita e in affitto hanno dimensioni medie molto diverse (rapporto fuori da 0,85-1,15).",
          "- affidabilita = 'bassa': meno di 3 vendite o 3 affitti con mq: media instabile.", "",
          vif_testo(Vt, "vendite"), vif_testo(At.rename(columns={"cpm": "ppm"}), "affitti"),
          "Con rapporti non si stima alcun modello: la multicollinearita' tra prezzo, mq e prezzo_mq non e' un problema per questa analisi. In un modello",
          "logaritmico le tre variabili sarebbero perfettamente collineari (log prezzo_mq = log prezzo - log mq): se ne possono includere solo due.", "",
          "Cautele: prezzi di annuncio; vendite troncate a 120.000 EUR (prezzo/mq sottostimato dove il tetto taglia gli immobili piu' grandi/cari);",
          "mq dichiarati dagli inserzionisti; annunci senza mq esclusi; esclusi IMU, imposte, manutenzione, sfitto."]
    open(os.path.join(out, "00_riepilogo_rendimento_mq.txt"), "w", encoding="utf-8").write("\n".join(R))

    # ---- grafici
    k = K.sort_values("rendimento_netto_mq_%")
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(k) + 1.5)))
    for i, (_, r) in enumerate(k.iterrows()):
        bassa = r["affidabilita"] == "bassa"
        kw = dict(alpha=0.4 if bassa else 0.9, hatch="//" if bassa else None, edgecolor="white")
        if pd.notna(r["rendimento_lordo_mq_%"]):
            ax.barh(i + 0.2, r["rendimento_lordo_mq_%"], height=0.38, color=BLU, **kw)
            ax.text(max(r["rendimento_lordo_mq_%"], 0), i + 0.2, " %.1f%%" % r["rendimento_lordo_mq_%"], va="center", fontsize=7)
        ax.barh(i - 0.2, r["rendimento_netto_mq_%"], height=0.38, color=ARANCIO, **kw)
        ax.text(max(r["rendimento_netto_mq_%"], 0), i - 0.2, " %.1f%%" % r["rendimento_netto_mq_%"], va="center", fontsize=7)
    ax.set_yticks(range(len(k))); ax.set_yticklabels(["%s (v=%d, a=%d)" % (c, nv, na) for c, nv, na in zip(k["citta"], k["n_vendite"], k["n_affitti"])], fontsize=8)
    ax.set_xlabel("rendimento annuo (%)")
    ax.set_title("%s - rendimento al metro quadro: lordo = canone + spese (blu), netto = solo canone (arancione)" % a.tipologia, loc="left", fontsize=10)
    ax.text(0.99, 0.02, "tratteggio = pochi annunci (<3 vendite o <3 affitti con mq)", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    stile(ax); salva(fig, os.path.join(out, "g_rendimento_mq.png"))

    k = K.sort_values("rendimento_netto_mq_%")
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(k) + 1.5)))
    for i, (_, r) in enumerate(k.iterrows()):
        ax.plot([r["rendimento_netto_senza_mq_%"], r["rendimento_netto_mq_%"]], [i, i], color=GRIGIO, linewidth=1.2)
        ax.plot(r["rendimento_netto_senza_mq_%"], i, "o", color=GRIGIO, markersize=6)
        ax.plot(r["rendimento_netto_mq_%"], i, "o", color=BLU, markersize=6)
    ax.set_yticks(range(len(k))); ax.set_yticklabels(k["citta"], fontsize=8)
    ax.set_xlabel("rendimento netto annuo = canone x 12 / prezzo (%)")
    ax.set_title("Effetto dei metri quadri: rendimento senza controllo dei mq (grigio) e al metro quadro (blu)", loc="left", fontsize=10)
    stile(ax); salva(fig, os.path.join(out, "g_confronto_con_senza_mq.png"))

    print("\n".join(R)); print("\nCreati: " + ", ".join(CREATI) + " | cartella: " + out)
    if SALTATI:
        print("Non creati:", "; ".join(SALTATI))
    return 0


if __name__ == "__main__":
    sys.exit(main())
