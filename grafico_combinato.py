# -*- coding: utf-8 -*-
"""
GDC Immobiliare - boxplot COMBINATO per citta': affitti (asse Y sinistro) e vendite (asse Y destro) in un unico grafico.

Input  (cartella analisi_modelli): output_vendite.csv, output_affitti.csv  (letti come gia' puliti, stessa preparazione di analisi_prezzi.py)
Output (cartella analisi_modelli/out_descr): boxplot_combinato_affitti_vendite_per_citta.png

Per ogni citta' due box affiancati:  blu = affitti (canone mensile senza spese, EUR/mese, scala a sinistra)  |  arancione = vendite (EUR, scala a destra).
Le due scale sono DIVERSE: confrontare la posizione dei box di una stessa serie tra citta', non l'altezza di un box blu con uno arancione.

Uso:   python grafico_combinato.py
       python grafico_combinato.py --affitti totale      (canone + spese condominiali; default: solo canone, senza spese)
       python grafico_combinato.py --min-n 5             (n minimo per disegnare un box; default 3)
Richiede analisi_prezzi.py nella stessa cartella. Nessun accesso a internet.
"""
import argparse, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import analisi_prezzi as ap

AQUI = os.path.dirname(os.path.abspath(__file__))
C_AFF, C_VEN, C_MED = "#2a78d6", "#e07b2e", "#12326a"


def disegna(ax, dati, pos, colore):
    bp = ax.boxplot(dati, positions=pos, widths=0.34, patch_artist=True, manage_ticks=False,
                    medianprops=dict(color=C_MED, linewidth=1.6),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=ap.GRIGIO, markeredgecolor=ap.GRIGIO))
    for b in bp["boxes"]:
        b.set(facecolor=colore, alpha=0.6, edgecolor=colore)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cartella", default="analisi_modelli")
    p.add_argument("--affitti", choices=["totale", "canone"], default="canone")
    p.add_argument("--min-n", type=int, default=3)
    a = p.parse_args()
    base = a.cartella if os.path.isabs(a.cartella) else os.path.join(AQUI, a.cartella)
    fv, fa = os.path.join(base, "output_vendite.csv"), os.path.join(base, "output_affitti.csv")
    for f in (fv, fa):
        if not os.path.exists(f):
            print("File non trovato:", f); return 1

    V = ap.prepara(fv, "vendita", a, [])
    A = ap.prepara(fa, "affitto", a, [])
    A["canone_totale"] = A["prezzo"] + A["spese_mese"]
    col_a, lab_a = ("canone_totale", "Affitti: canone + spese condominiali (EUR/mese)") if a.affitti == "totale" \
        else ("prezzo", "Affitti: canone mensile (EUR/mese)")
    A = A.dropna(subset=[col_a])

    gv = {c: g["prezzo"].to_numpy() for c, g in V.groupby("citta") if len(g) >= a.min_n}
    ga = {c: g[col_a].to_numpy() for c, g in A.groupby("citta") if len(g) >= a.min_n}
    import numpy as np
    med_a = {c: np.median(x) for c, x in ga.items()}
    med_v = {c: np.median(x) for c, x in gv.items()}
    citta = sorted(med_a, key=lambda c: -med_a[c]) + sorted([c for c in med_v if c not in med_a], key=lambda c: -med_v[c])
    if not citta:
        print("Nessuna citta' con almeno", a.min_n, "annunci."); return 1

    fig, ax1 = plt.subplots(figsize=(max(9, 1.0 * len(citta) + 3), 6))
    ax2 = ax1.twinx()
    pa = [i - 0.19 for i, c in enumerate(citta) if c in ga]
    pv = [i + 0.19 for i, c in enumerate(citta) if c in gv]
    disegna(ax1, [ga[c] for c in citta if c in ga], pa, C_AFF)
    disegna(ax2, [gv[c] for c in citta if c in gv], pv, C_VEN)

    ax1.set_xlim(-0.6, len(citta) - 0.4)
    ax1.set_xticks(range(len(citta)))
    ax1.set_xticklabels(["%s\n(aff. n=%s | vend. n=%s)" % (c, len(ga[c]) if c in ga else "-", len(gv[c]) if c in gv else "-")
                         for c in citta], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel(lab_a, color=C_AFF); ax1.tick_params(axis="y", colors=C_AFF)
    ax2.set_ylabel("Vendite: prezzo di vendita (EUR)", color=C_VEN); ax2.tick_params(axis="y", colors=C_VEN)
    ax1.set_ylim(bottom=0); ax2.set_ylim(bottom=0)
    ax1.set_title("Affitti (asse sinistro) e vendite (asse destro) per citta'", loc="left", fontsize=11)
    ax1.set_facecolor(ap.FONDO)
    ax1.grid(axis="y", color="#e6e6e3", linewidth=0.7); ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
    ax1.legend(handles=[Patch(facecolor=C_AFF, alpha=0.6, edgecolor=C_AFF, label="Affitti (scala sinistra)"),
                        Patch(facecolor=C_VEN, alpha=0.6, edgecolor=C_VEN, label="Vendite (scala destra)")],
               loc="upper right", frameon=False, fontsize=9)

    out = os.path.join(base, "out_descr"); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "boxplot_combinato_affitti_vendite_per_citta.png")
    ap.salva_fig(fig, path)
    print("Salvato:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
