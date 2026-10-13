#!/usr/bin/env python3
"""
grafici_boxplot.py - boxplot per citta' dei file annunci_vendita.csv e annunci_affitto.csv.

Per ogni file crea un PNG con due pannelli affiancati, una riga per citta':
  - vendita : prezzo (EUR)            e   prezzo al mq (EUR/mq)
  - affitto : canone (EUR/mese)       e   canone al mq (EUR/mq al mese)
Ogni annuncio e' mostrato come punto sopra il boxplot; tra parentesi il numero di annunci (n).
Con meno di 3 annunci la citta' mostra solo i punti e la mediana (un box non avrebbe senso).
La citta' e' quella dell'area Idealista abbinata (colonna citta_area); se l'abbinamento manca si usa
il nome del file HTML salvato (es. PISA.html -> Pisa).

Uso:
    python grafici_boxplot.py
    python grafici_boxplot.py --vendita annunci_vendita.csv --affitto annunci_affitto.csv
Richiede matplotlib (incluso in Anaconda). Output: boxplot_vendita.png, boxplot_affitto.png
"""
import argparse, csv, io, os, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE = os.path.dirname(os.path.abspath(__file__))
SUPERFICIE, SERIE, INK, INK2, GRIGLIA = "#fcfcfb", "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"


def leggi(p):
    raw = open(p, "rb").read()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return list(csv.DictReader(io.StringIO(raw.decode(enc)), delimiter=";"))
        except UnicodeDecodeError:
            continue
    return []


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def citta(r):
    if r.get("citta_area"):
        return r["citta_area"].split("|")[0]
    f = os.path.splitext(r.get("file_sorgente", ""))[0].strip()
    return f.title() if f else "(non abbinata)"


def migliaia(v, _=None):
    return "{:,.0f}".format(v).replace(",", ".")


def pannello(ax, dati, titolo, unita):
    """dati: {citta: [valori]}; una riga per citta', ordinate per mediana."""
    ordine = sorted(dati, key=lambda c: statistics.median(dati[c]))
    ax.set_facecolor(SUPERFICIE)
    for i, c in enumerate(ordine):
        v = sorted(dati[c])
        if len(v) >= 3:
            ax.boxplot([v], positions=[i], vert=False, widths=0.5, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=SERIE + "33", edgecolor=SERIE, linewidth=1.5),
                       whiskerprops=dict(color=SERIE, linewidth=1.5), capprops=dict(color=SERIE, linewidth=1.5),
                       medianprops=dict(color=SERIE, linewidth=2.5))
        else:
            m = statistics.median(v)
            ax.plot([m, m], [i - 0.25, i + 0.25], color=SERIE, linewidth=2.5)
        j = [((k * 0.37) % 1 - 0.5) * 0.36 for k in range(len(v))]    # sfalsamento fisso, riproducibile
        ax.scatter(v, [i + d for d in j], s=40, color=SERIE, edgecolor=SUPERFICIE, linewidth=1.5, zorder=3)
    ax.set_yticks(range(len(ordine)))
    ax.set_yticklabels(["%s  (n=%d)" % (c, len(dati[c])) for c in ordine], color=INK, fontsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(migliaia))
    ax.tick_params(axis="x", colors=INK2, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(unita, color=INK2, fontsize=9)
    ax.set_title(titolo, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.grid(axis="x", color=GRIGLIA, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRIGLIA)
    ax.set_ylim(-0.7, len(ordine) - 0.3)


def grafico(righe, nome_out, tipo):
    prezzi, mq = {}, {}
    for r in righe:
        p, m = num(r.get("prezzo")), num(r.get("prezzo_mq"))
        c = citta(r)
        if p is not None:
            prezzi.setdefault(c, []).append(p)
        if m is not None:
            mq.setdefault(c, []).append(m)
    if not prezzi:
        print("Nessun annuncio con prezzo per %s: grafico non creato." % tipo)
        return False
    vend = tipo == "vendita"
    n_tot = sum(len(v) for v in prezzi.values())
    h = max(3.6, 0.55 * len(prezzi) + 1.9)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, h), facecolor=SUPERFICIE)
    pannello(a1, prezzi, "Prezzo di vendita" if vend else "Canone mensile", "€" if vend else "€ al mese")
    pannello(a2, mq, "Prezzo al mq" if vend else "Canone al mq", "€ al mq" if vend else "€ al mq al mese")
    fig.suptitle("Annunci in %s per città - %d annunci, %d città" % (tipo, n_tot, len(prezzi)),
                 x=0.01, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.text(0.01, 0.005, "Box: dal 1° al 3° quartile, linea = mediana, baffi = minimo e massimo; punti = singoli annunci. "
             "Con meno di 3 annunci: solo punti e mediana.", color=INK2, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(BASE, nome_out), dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    print("Creato %s (%d annunci, %d citta')" % (nome_out, n_tot, len(prezzi)))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendita", default="annunci_vendita.csv")
    ap.add_argument("--affitto", default="annunci_affitto.csv")
    a = ap.parse_args()
    for tipo, f in (("vendita", a.vendita), ("affitto", a.affitto)):
        p = f if os.path.isabs(f) else os.path.join(BASE, f)
        if not os.path.exists(p):
            print("File %s non trovato: salto %s." % (f, tipo))
            continue
        grafico(leggi(p), "boxplot_%s.png" % tipo, tipo)


if __name__ == "__main__":
    sys.exit(main())
