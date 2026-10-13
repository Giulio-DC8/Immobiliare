# -*- coding: utf-8 -*-
"""Esporta in un file .txt i link delle schede (uno per riga) da annunci_vendita.csv / annunci_affitto.csv.
Nessun accesso a internet: legge solo i CSV locali.

Esempi:
  python esporta_link.py
  python esporta_link.py --operazione vendita --citta Pisa --max-prezzo-mq 2000
  python esporta_link.py --out link_vendita.txt
"""
import argparse, csv, os, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
FILE = {"vendita": "annunci_vendita.csv", "affitto": "annunci_affitto.csv"}

def num(v):
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--operazione", choices=["vendita", "affitto", "entrambe"], default="entrambe")
    p.add_argument("--citta", action="append")
    p.add_argument("--max-prezzo", type=float); p.add_argument("--min-prezzo", type=float)
    p.add_argument("--max-prezzo-mq", type=float); p.add_argument("--min-prezzo-mq", type=float)
    p.add_argument("--max-mq", type=float); p.add_argument("--min-mq", type=float)
    p.add_argument("--solo-ribassati", action="store_true")
    p.add_argument("--solo-presenti", action="store_true")
    p.add_argument("--ordina", choices=["prezzo_mq", "prezzo", "mq", "citta", "nessuno"], default="nessuno")
    p.add_argument("--limite", type=int)
    p.add_argument("--out", default="link_schede.txt")
    a = p.parse_args()

    ops = ["vendita", "affitto"] if a.operazione == "entrambe" else [a.operazione]
    citta = [c.lower() for c in a.citta] if a.citta else None
    righe = []
    for op in ops:
        f = os.path.join(AQUI, FILE[op])
        if not os.path.exists(f):
            print("File assente:", FILE[op]); continue
        with open(f, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh, delimiter=";"):
                pr, mq, pmq = num(r["prezzo"]), num(r["mq"]), num(r["prezzo_mq"])
                if citta and (r.get("citta_area") or "").lower() not in citta: continue
                if a.max_prezzo is not None and (pr is None or pr > a.max_prezzo): continue
                if a.min_prezzo is not None and (pr is None or pr < a.min_prezzo): continue
                if a.max_prezzo_mq is not None and (pmq is None or pmq > a.max_prezzo_mq): continue
                if a.min_prezzo_mq is not None and (pmq is None or pmq < a.min_prezzo_mq): continue
                if a.max_mq is not None and (mq is None or mq > a.max_mq): continue
                if a.min_mq is not None and (mq is None or mq < a.min_mq): continue
                if a.solo_ribassati and not (r.get("ribasso_pct") or "").strip(): continue
                if a.solo_presenti and r.get("presente_ultimo_lancio") == "no": continue
                righe.append((r, pr, mq, pmq))
    chiavi = {"prezzo_mq": lambda x: (x[3] is None, x[3] or 0), "prezzo": lambda x: (x[1] is None, x[1] or 0),
              "mq": lambda x: (x[2] is None, x[2] or 0), "citta": lambda x: (x[0].get("citta_area") or "~").lower()}
    if a.ordina != "nessuno":
        righe.sort(key=chiavi[a.ordina])
    if a.limite:
        righe = righe[:a.limite]
    visti, link = set(), []
    for r, *_ in righe:
        u = "https://www.idealista.it/immobile/%s/" % r["id_annuncio"]
        if u not in visti:
            visti.add(u); link.append(u)
    out = a.out if os.path.isabs(a.out) else os.path.join(AQUI, a.out)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(link) + ("\n" if link else ""))
    print("Scritti %d link in %s" % (len(link), out))

if __name__ == "__main__":
    sys.exit(main())
