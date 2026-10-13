# -*- coding: utf-8 -*-
"""
GDC Immobiliare - tabella aggregata per CITTA' x TIPOLOGIA.

Input  (cartella analisi_modelli): output_vendite.csv, output_affitti.csv  (considerati GIA' PULITI: nessuna esclusione di
        duplicati, annunci rif=NO o valori estremi; si scartano solo righe vuote e righe senza prezzo valido)
Output (cartella analisi_modelli): output_aggregato.csv   (separatore ';', codifica utf-8-sig, apribile in Excel)

Una riga per ogni combinazione citta' x tipologia (Monolocale, Bilocale, Trilocale, Quadrilocale+, Altro (attico/casa)); citta' NON accorpate.
Colonne:
  citta, tipologia
  n_vendite, prezzo_vendita_medio          media aritmetica del prezzo di vendita (EUR)
  n_affitti, canone_medio                  media aritmetica del canone mensile (EUR/mese)
  n_spese, spese_condominiali_medie        media delle spese condominiali (EUR/mese) calcolata sull'INSIEME di vendite e affitti
                                           (la spesa e' la stessa per chi compra e per chi affitta); le spese mancanti non entrano
                                           nella media, n_spese indica su quanti annunci e' calcolata.
Le celle senza dati restano vuote (n = 0). Le medie su 1-2 annunci sono poco affidabili: guardare sempre le numerosita'.

Uso:   python aggrega_output.py
       python aggrega_output.py --cartella altra_cartella
Richiede analisi_prezzi.py nella stessa cartella. Nessun accesso a internet.
"""
import argparse, os, sys
import pandas as pd
import analisi_prezzi as ap

AQUI = os.path.dirname(os.path.abspath(__file__))
ORDINE_TIP = ["Monolocale", "Bilocale", "Trilocale", "Quadrilocale+", "Altro (attico/casa)"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cartella", default="analisi_modelli")
    a = p.parse_args()
    base = a.cartella if os.path.isabs(a.cartella) else os.path.join(AQUI, a.cartella)
    fv, fa = os.path.join(base, "output_vendite.csv"), os.path.join(base, "output_affitti.csv")
    for f in (fv, fa):
        if not os.path.exists(f):
            print("File non trovato:", f); return 1

    rap = []
    V = ap.prepara(fv, "vendita", a, rap)
    A = ap.prepara(fa, "affitto", a, rap)
    print("\n".join(rap))

    chiave = ["citta", "tip"]
    v = V.groupby(chiave)["prezzo"].agg(n_vendite="count", prezzo_vendita_medio="mean")
    f = A.groupby(chiave)["prezzo"].agg(n_affitti="count", canone_medio="mean")
    sp = pd.concat([V[chiave + ["spese_mese"]], A[chiave + ["spese_mese"]]]).dropna(subset=["spese_mese"])
    s = sp.groupby(chiave)["spese_mese"].agg(n_spese="count", spese_condominiali_medie="mean")

    T = v.join(f, how="outer").join(s, how="outer").reset_index().rename(columns={"tip": "tipologia"})
    for c in ("n_vendite", "n_affitti", "n_spese"):
        T[c] = T[c].fillna(0).astype(int)
    T["_o"] = T["tipologia"].map({t: i for i, t in enumerate(ORDINE_TIP)}).fillna(99)
    T = T.sort_values(["citta", "_o"]).drop(columns="_o")
    T = T[["citta", "tipologia", "n_vendite", "prezzo_vendita_medio", "n_affitti", "canone_medio",
           "n_spese", "spese_condominiali_medie"]]
    for c in ("prezzo_vendita_medio", "canone_medio", "spese_condominiali_medie"):
        T[c] = T[c].round(2)

    path = os.path.join(base, "output_aggregato.csv")
    ap.scrivi_csv(T, path)
    print("\nRighe (citta' x tipologia): %d | citta': %d" % (len(T), T["citta"].nunique()))
    print("Annunci vendita: %d | affitto: %d | con spese indicate: %d" % (len(V), len(A), int(T["n_spese"].sum())))
    print("Salvato:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
