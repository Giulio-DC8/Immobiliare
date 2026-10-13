#!/usr/bin/env python3
"""
filtra_citta.py - genera citta.txt dalle province con più studenti universitari fuori regione.

Legge province_iscritti.csv (provincia;iscritti_fuori_regione), tiene le province con
iscritti >= soglia e scrive citta.txt con i comuni da cercare: capoluogo/capoluoghi più i
comuni non capoluogo con grandi poli sanitari-universitari della stessa provincia
(es. Rozzano e Segrate per Milano).

Uso:
    python filtra_citta.py                  # soglia 1000
    python filtra_citta.py --soglia 3000    # più selettivo
    python filtra_citta.py --soglia 0       # tutte le province
    python filtra_citta.py --tabella        # solo il riepilogo per soglie, senza scrivere
"""
import argparse, csv, os

BASE = os.path.dirname(os.path.abspath(__file__))

# provincia (come nel file dati) -> comuni da cercare. Se assente, vale il nome della provincia.
COMUNI = {
    "PESARO E URBINO": ["Pesaro", "Urbino"],
    "FORLI'-CESENA": ["Forlì", "Cesena"],
    "MONZA E BRIANZA": ["Monza"],
    "BARLETTA - ANDRIA - TRANI": ["Barletta", "Andria", "Trani"],
    "OLBIA-TEMPIO": ["Olbia"],
    "VERBANO CUSIO OSSOLA": ["Verbania"],
    # comuni non capoluogo con poli rilevanti
    "MILANO": ["Milano", "Rozzano", "Segrate", "Pieve Emanuele"],
    "TORINO": ["Torino", "Orbassano"],
    "FERRARA": ["Ferrara", "Cona"],
    "VERONA": ["Verona", "Negrar"],
    "CAGLIARI": ["Cagliari", "Monserrato"],
    "SALERNO": ["Salerno", "Baronissi", "Fisciano"],
    "COSENZA": ["Cosenza", "Rende"],
    "FROSINONE": ["Frosinone", "Cassino"],
}

def titolo(p):
    return p.title().replace("L'Aquila", "L'Aquila")

def leggi():
    with open(os.path.join(BASE, "province_iscritti.csv"), encoding="utf-8", newline="") as f:
        return [(r["provincia"], int(r["iscritti_fuori_regione"]))
                for r in csv.DictReader(f, delimiter=";")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soglia", type=int, default=1000)
    ap.add_argument("--tabella", action="store_true")
    a = ap.parse_args()
    dati = leggi()
    totale = sum(n for _, n in dati)
    if a.tabella:
        print("soglia   province  comuni  quota iscritti coperta")
        for s in (0, 500, 1000, 2000, 3000, 5000, 10000):
            sel = [(p, n) for p, n in dati if n >= s]
            com = sum(len(COMUNI.get(p, [titolo(p)])) for p, _ in sel)
            print("%6d   %8d  %6d  %5.1f%%" % (s, len(sel), com, 100.0 * sum(n for _, n in sel) / totale))
        return
    sel = [(p, n) for p, n in dati if n >= a.soglia]
    righe = ["# Generato da filtra_citta.py (soglia: %d iscritti fuori regione per provincia)." % a.soglia,
             "# Per cambiare l'elenco rilancia: python filtra_citta.py --soglia N",
             "# Le righe con # vengono ignorate dagli script."]
    for p, n in sel:
        righe.append("# %s (%d)" % (p, n))
        righe += COMUNI.get(p, [titolo(p)])
    with open(os.path.join(BASE, "citta.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(righe) + "\n")
    n_com = sum(1 for r in righe if not r.startswith("#"))
    print("citta.txt: %d province, %d comuni (soglia %d)" % (len(sel), n_com, a.soglia))

if __name__ == "__main__":
    main()
