# -*- coding: utf-8 -*-
"""
GDC Immobiliare - classifica del RENDIMENTO per citta' x tipologia e per citta' (tutte le tipologie insieme).

Input : analisi_modelli/output_aggregato.csv   (prodotto da aggrega_output.py)
Output: analisi_modelli/out_rendimento/

Definizioni (valori medi di cella):
  rendimento_lordo_%   = (canone_medio + spese_condominiali_medie) * 12 / prezzo_vendita_medio * 100
  rendimento_netto_%   = canone_medio * 12 / prezzo_vendita_medio * 100
                         (le spese condominiali sono pagate dall'inquilino in aggiunta al canone: il lordo le include, il netto no;
                          non include IMU, imposte, manutenzione, periodi sfitti)
  incidenza_spese_%    = spese_condominiali_medie / canone_medio * 100
Nessuna soglia minima: entrano tutte le celle che hanno almeno 1 annuncio di vendita E almeno 1 di affitto.
La colonna affidabilita' segnala le celle con poche osservazioni: "alta" se vendite>=3 e affitti>=3, altrimenti "bassa".
Se la cella non ha spese rilevate (n_spese = 0), il rendimento lordo e l'incidenza spese restano vuoti (rendimento netto calcolato).
Livello citta': medie ponderate sul numero di annunci di tutte le tipologie della citta' (equivalenti alle medie sugli annunci).

ATTENZIONE
  - I prezzi di vendita sono troncati a 120.000 EUR dal filtro di ricerca: il prezzo medio e' sottostimato nelle citta' piu' care
    e il rendimento tende a risultare sovrastimato.
  - Il confronto e' tra medie di annunci diversi (non lo stesso immobile in vendita e in affitto) e senza controllo dei metri quadri.
  - Le medie su 1-2 annunci sono casuali: usare la colonna affidabilita'.

Uso:   python classifica_rendimento.py
Dipendenze: pandas, numpy, matplotlib (incluse in Anaconda). Nessun accesso a internet.
"""
import argparse, io, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
BLU, ARANCIO, GRIGIO, FONDO = "#2a78d6", "#e07b2e", "#8a8f98", "#fcfcfb"
ORDINE_TIP = ["Monolocale", "Bilocale", "Trilocale", "Quadrilocale+", "Altro (attico/casa)"]


def leggi(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            t = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    p = t.splitlines()[0]
    return pd.read_csv(io.StringIO(t), sep=";" if p.count(";") >= p.count(",") else ",")


def scrivi_csv(df, path):
    df.to_csv(path, sep=";", index=False, encoding="utf-8-sig")


def indicatori(T):
    T["rendimento_lordo_%"] = (T["canone_medio"] + T["spese_condominiali_medie"]) * 12 / T["prezzo_vendita_medio"] * 100
    T["rendimento_netto_%"] = T["canone_medio"] * 12 / T["prezzo_vendita_medio"] * 100
    T["incidenza_spese_%"] = T["spese_condominiali_medie"] / T["canone_medio"] * 100
    T["affidabilita"] = np.where((T["n_vendite"] >= 3) & (T["n_affitti"] >= 3), "alta", "bassa")
    return T


def stile(ax):
    ax.set_facecolor(FONDO); ax.grid(color="#e6e6e3", linewidth=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def dispersione(T, etichetta, titolo, path, n_etichette, ycol="rendimento_netto_%"):
    d = T.dropna(subset=["spese_condominiali_medie", ycol]).copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for aff, col, mk in (("alta", BLU, "o"), ("bassa", ARANCIO, "^")):
        s = d[d.affidabilita == aff]
        ax.scatter(s["spese_condominiali_medie"], s[ycol], s=30 + 6 * np.sqrt(s[["n_vendite", "n_affitti"]].min(axis=1)),
                   color=col, alpha=0.65, marker=mk, edgecolor="none",
                   label="affidabilita' alta (n>=3 vendite e affitti)" if aff == "alta" else "affidabilita' bassa (pochi annunci)")
    top = d.sort_values(ycol, ascending=False).head(n_etichette)
    for _, r in top.iterrows():
        ax.annotate(r[etichetta], (r["spese_condominiali_medie"], r[ycol]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("spese condominiali medie (EUR/mese)  -  meglio a sinistra")
    ax.set_ylabel(("rendimento lordo = (canone + spese) x 12 / prezzo (%)" if ycol == "rendimento_lordo_%" else "rendimento netto = canone x 12 / prezzo (%)") + "  -  meglio in alto")
    ax.set_title(titolo + "  (etichettate le %d con miglior rendimento %s)" % (n_etichette, "lordo" if ycol == "rendimento_lordo_%" else "netto"), loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8); stile(ax)
    fig.patch.set_facecolor(FONDO); fig.tight_layout(); fig.savefig(path, dpi=140, facecolor=FONDO); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cartella", default="analisi_modelli")
    a = ap.parse_args()
    base = a.cartella if os.path.isabs(a.cartella) else os.path.join(AQUI, a.cartella)
    f_in = os.path.join(base, "output_aggregato.csv")
    if not os.path.exists(f_in):
        print("File non trovato:", f_in, "(esegui prima aggrega_output.py)"); return 1
    out = os.path.join(base, "out_rendimento"); os.makedirs(out, exist_ok=True)

    D = leggi(f_in)
    for c in ("n_vendite", "n_affitti", "n_spese", "prezzo_vendita_medio", "canone_medio", "spese_condominiali_medie"):
        D[c] = pd.to_numeric(D[c], errors="coerce")
    for c in ("n_vendite", "n_affitti", "n_spese"):
        D[c] = D[c].fillna(0)
    ok = (D["n_vendite"] >= 1) & (D["n_affitti"] >= 1) & D["prezzo_vendita_medio"].notna() & D["canone_medio"].notna()
    esc = D[~ok]
    C = indicatori(D[ok].copy())
    C["_o"] = C["tipologia"].map({t: i for i, t in enumerate(ORDINE_TIP)}).fillna(99)
    C = C.sort_values("rendimento_netto_%", ascending=False, na_position="last").drop(columns="_o")
    colc = ["citta", "tipologia", "n_vendite", "prezzo_vendita_medio", "n_affitti", "canone_medio", "n_spese",
            "spese_condominiali_medie", "rendimento_lordo_%", "rendimento_netto_%", "incidenza_spese_%", "affidabilita"]
    C = C[colc].round(2)
    scrivi_csv(C, os.path.join(out, "rendimento_celle_citta_tipologia.csv"))
    scrivi_csv(esc[["citta", "tipologia", "n_vendite", "n_affitti"]], os.path.join(out, "celle_senza_vendite_o_affitti.csv"))

    # ---- livello citta' (medie ponderate sul numero di annunci)
    righe = []
    for citta, g in D.groupby("citta"):
        nv, na, ns = g["n_vendite"].sum(), g["n_affitti"].sum(), g["n_spese"].sum()
        pv = (g["prezzo_vendita_medio"].fillna(0) * g["n_vendite"]).sum() / nv if nv > 0 else np.nan
        cn = (g["canone_medio"].fillna(0) * g["n_affitti"]).sum() / na if na > 0 else np.nan
        sp = (g["spese_condominiali_medie"].fillna(0) * g["n_spese"]).sum() / ns if ns > 0 else np.nan
        righe.append({"citta": citta, "n_vendite": int(nv), "prezzo_vendita_medio": pv, "n_affitti": int(na), "canone_medio": cn,
                      "n_spese": int(ns), "spese_condominiali_medie": sp})
    K = pd.DataFrame(righe)
    K = K[(K.n_vendite >= 1) & (K.n_affitti >= 1)].copy()
    K = indicatori(K).sort_values("rendimento_netto_%", ascending=False, na_position="last")
    colk = [c for c in colc if c != "tipologia"]
    K = K[colk].round(2)
    scrivi_csv(K, os.path.join(out, "rendimento_citta.csv"))

    Cg = indicatori(D[ok].copy()); Cg["etichetta"] = Cg["citta"] + " " + Cg["tipologia"].str.replace(" (attico/casa)", "", regex=False)
    dispersione(Cg, "etichetta", "Citta' x tipologia: rendimento netto e spese condominiali", os.path.join(out, "dispersione_celle.png"), 12)
    Kg = indicatori(K.copy()); Kg["etichetta"] = Kg["citta"]
    dispersione(Kg, "etichetta", "Citta': rendimento netto e spese condominiali", os.path.join(out, "dispersione_citta.png"), len(Kg))
    # solo BILOCALE (il budget di circa 120.000 EUR corrisponde in media all'acquisto di un bilocale): un punto per citta'
    Bg = Cg[Cg["tipologia"] == "Bilocale"].copy(); Bg["etichetta"] = Bg["citta"]
    dispersione(Bg, "etichetta", "Citta' - solo bilocale: rendimento netto e spese condominiali", os.path.join(out, "dispersione_citta_bilocale.png"), len(Bg))
    dispersione(Bg, "etichetta", "Citta' - solo bilocale: rendimento lordo (canone + spese) e spese condominiali", os.path.join(out, "dispersione_citta_bilocale_lordo.png"), len(Bg), "rendimento_lordo_%")

    # ---- riepilogo testuale
    R = ["CLASSIFICA RENDIMENTO (lordo = canone + spese; netto = solo canone; spese pagate dall'inquilino)",
         "Celle citta' x tipologia con vendite e affitti: %d | escluse perche' senza vendite o senza affitti: %d (celle_senza_vendite_o_affitti.csv)" % (len(C), len(esc)),
         "Nessuna soglia minima di annunci: guardare la colonna affidabilita' (alta = almeno 3 vendite e 3 affitti).",
         "", "=" * 78, "PER CITTA' (tutte le tipologie insieme) - ordinate per rendimento netto", "=" * 78,
         K[["citta", "n_vendite", "n_affitti", "prezzo_vendita_medio", "canone_medio", "spese_condominiali_medie",
            "rendimento_lordo_%", "rendimento_netto_%", "incidenza_spese_%"]].to_string(index=False), "",
         "=" * 78, "PER CITTA' x TIPOLOGIA - prime 15 per rendimento netto (affidabilita' ALTA)", "=" * 78,
         C[C.affidabilita == "alta"].head(15)[["citta", "tipologia", "n_vendite", "n_affitti", "prezzo_vendita_medio", "canone_medio",
                                             "spese_condominiali_medie", "rendimento_lordo_%", "rendimento_netto_%"]].to_string(index=False), "",
         "Celle con le spese condominiali piu' basse (affidabilita' ALTA, prime 10):",
         C[(C.affidabilita == "alta") & C.spese_condominiali_medie.notna()].sort_values("spese_condominiali_medie").head(10)[
             ["citta", "tipologia", "spese_condominiali_medie", "rendimento_lordo_%", "rendimento_netto_%"]].to_string(index=False), "",
         "Cautele: prezzi di vendita troncati a 120.000 EUR (rendimenti sovrastimati nelle citta' care); confronto tra annunci diversi",
         "senza metri quadri; medie su pochi annunci instabili; esclusi IMU, imposte, manutenzione, sfitto."]
    open(os.path.join(out, "00_riepilogo_rendimento.txt"), "w", encoding="utf-8").write("\n".join(R))
    print("\n".join(R)); print("\nSalvato in:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
