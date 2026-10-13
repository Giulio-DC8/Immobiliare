#!/usr/bin/env python3
"""
grafici_boxplot_affitto.py - boxplot per citta' dei canoni in affitto (annunci_affitto.csv).

Crea boxplot_affitto.png con due pannelli: canone mensile (EUR/mese) e canone al mq (EUR/mq al mese),
una riga per citta', ordinate per mediana. Ogni annuncio e' un punto; (n) = numero di annunci.
Con meno di 3 annunci la citta' mostra solo i punti e la mediana.

Uso:
    python grafici_boxplot_affitto.py
    python grafici_boxplot_affitto.py --file annunci_affitto.csv --solo-ultimo-lancio
    python grafici_boxplot_affitto.py --max-mq 60      # esclude annunci oltre 60 mq (se presenti)

Prima va aggiornato annunci_affitto.csv:  python leggi_annunci.py
La grafica e' condivisa con grafici_boxplot.py (stesso stile, stesse regole).
"""
import argparse, os, sys
import grafici_boxplot as g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="annunci_affitto.csv")
    ap.add_argument("--solo-ultimo-lancio", action="store_true",
                    help="usa solo gli annunci presenti nelle pagine dell'ultimo lancio di leggi_annunci.py")
    ap.add_argument("--max-mq", type=float, default=0, help="esclude gli annunci con superficie maggiore (0 = nessun limite)")
    a = ap.parse_args()
    p = a.file if os.path.isabs(a.file) else os.path.join(g.BASE, a.file)
    if not os.path.exists(p):
        sys.exit("File %s non trovato: lancia prima  python leggi_annunci.py" % a.file)
    righe = g.leggi(p)
    if a.solo_ultimo_lancio:
        righe = [r for r in righe if r.get("presente_ultimo_lancio", "sì") == "sì"]
    if a.max_mq:
        righe = [r for r in righe if (g.num(r.get("mq")) or 0) <= a.max_mq]
    if not g.grafico(righe, "boxplot_affitto.png", "affitto"):
        print("Nessun annuncio in affitto: salvare le pagine in affitto in dati\\pagine\\ e rilanciare leggi_annunci.py.")


if __name__ == "__main__":
    sys.exit(main())
