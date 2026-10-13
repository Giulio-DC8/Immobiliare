# -*- coding: utf-8 -*-
"""Estrae le «Spese condominiali» dalle schede di dettaglio che hai salvato tu in dati\\pagine\\pagina_ID
(file ID.html oppure pagina_ID.html). Nessun accesso a internet: legge solo file locali.

Output:
  spese_condominiali.csv        id_annuncio;spese_mese;stato;file
  annunci_vendita_spese.csv     annunci_vendita.csv + spese_mese, spese_anno
  annunci_affitto_spese.csv     annunci_affitto.csv + spese_mese, spese_anno
  (in tutti e due i file anche latitudine, longitudine: centro della mappa nella scheda, decimali con la virgola; vuote se la scheda manca)

stato: ok | nessuna_spesa | non_indicata | pagina_bloccata | id_diverso | illeggibile

Uso:
  python leggi_spese.py
  python leggi_spese.py --cartella dati\\pagine\\pagina_ID
"""
import argparse, csv, html, os, re, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
NOME = re.compile(r"^(?:pagina_)?(\d+)(\.[A-Za-z0-9]+)?$", re.I)
RE_SPESE = re.compile(r"Spese\s+condominiali\s*:?\s*([^<]{0,60})", re.I)
RE_IMPORTO = re.compile(r"(\d[\d\.]*)(?:,(\d{1,2}))?\s*(?:€|euro|eur)\s*(?:/|al|per)?\s*(mese|anno)?", re.I)
RE_CANON = re.compile(r"/immobile/(\d+)", re.I)
RE_CENTRO = re.compile(r"center=(-?\d{1,3}\.\d+)(?:%2C|%2c|,)(-?\d{1,3}\.\d+)")   # centro della mappa nella scheda: lat,lon
BLOCCO = re.compile(r"captcha-delivery|datadome|geo\.captcha|access denied|accesso negato", re.I)


def leggi(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", "replace")


def importo(testo):
    m = RE_IMPORTO.search(testo)
    if not m:
        return None
    v = float(m.group(1).replace(".", "") + ("." + m.group(2) if m.group(2) else ""))
    if (m.group(3) or "mese").lower() == "anno":
        v = v / 12.0
    return round(v, 2)


def analizza(testo, id_file):
    canon = RE_CANON.search(testo)
    if canon and canon.group(1) != id_file:
        return None, "id_diverso"
    senza = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", testo)
    piano = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", senza)))
    trovate = [m.group(1) for m in RE_SPESE.finditer(piano)]
    if not trovate:
        if BLOCCO.search(testo) and len(testo) < 50000:
            return None, "pagina_bloccata"
        return None, "non_indicata"
    for resto in trovate:
        v = importo(resto)
        if v is not None:
            return v, "ok"
    if any(re.search(r"nessun|assent|non\s+presenti", r, re.I) for r in trovate):
        return 0.0, "nessuna_spesa"
    return None, "non_indicata"


def fmt(v):
    return "" if v is None else ("%.2f" % v).replace(".", ",")


def coordinate(testo):
    """(lat, lon) dal parametro center= della mappa nella scheda; None se assente o fuori dall'Italia."""
    m = RE_CENTRO.search(testo)
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    return (lat, lon) if (35.0 <= lat <= 48.0 and 6.0 <= lon <= 19.0) else None


def fmt_coord(c, i):
    return "" if c is None else ("%.7f" % c[i]).replace(".", ",")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cartella", default=os.path.join("dati", "pagine", "pagina_ID"))
    p.add_argument("--out", default="spese_condominiali.csv")
    a = p.parse_args()
    cart = a.cartella if os.path.isabs(a.cartella) else os.path.join(AQUI, a.cartella)
    if not os.path.isdir(cart):
        print("Cartella non trovata:", cart); return 1

    righe, cont, coord = {}, {}, {}
    for nome in sorted(os.listdir(cart)):
        m = NOME.match(nome)
        if not m or os.path.isdir(os.path.join(cart, nome)):
            continue
        idf = m.group(1)
        try:
            testo = leggi(os.path.join(cart, nome))
            v, stato = analizza(testo, idf)
            coord[idf] = coordinate(testo)
        except Exception:
            v, stato = None, "illeggibile"
        righe[idf] = (v, stato, nome)
        cont[stato] = cont.get(stato, 0) + 1

    out = os.path.join(AQUI, a.out)
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["id_annuncio", "spese_mese", "stato", "file"])
        for idf, (v, stato, nome) in righe.items():
            w.writerow([idf, fmt(v), stato, nome])
    print("Schede lette: %d  ->  %s" % (len(righe), out))
    for k, n in sorted(cont.items()):
        print("  %-16s %d" % (k, n))
    print("Schede con coordinate (mappa): %d su %d" % (sum(1 for c in coord.values() if c), len(righe)))

    rif = {}
    fr = os.path.join(AQUI, "classificazione_rif.csv")
    if os.path.exists(fr):
        with open(fr, encoding="utf-8-sig", newline="") as fh:
            for x in csv.DictReader(fh, delimiter=";"):
                rif[(x["operazione"], x["id_annuncio"])] = x

    gem = {}
    fg = os.path.join(AQUI, "classificazioni_llm.csv")
    if os.path.exists(fg):
        with open(fg, encoding="utf-8-sig", newline="") as fh:
            for x in csv.DictReader(fh, delimiter=";"):
                gem[(x["operazione"], x["id_annuncio"])] = x

    grk = {}
    fk = os.path.join(AQUI, "classificazioni_grok.csv")
    if os.path.exists(fk):
        with open(fk, encoding="utf-8-sig", newline="") as fh:
            for x in csv.DictReader(fh, delimiter=";"):
                grk[(x["operazione"], x["id_annuncio"])] = x

    oai = {}
    fo = os.path.join(AQUI, "classificazioni_openai.csv")
    if os.path.exists(fo):
        with open(fo, encoding="utf-8-sig", newline="") as fh:
            for x in csv.DictReader(fh, delimiter=";"):
                oai[(x["operazione"], x["id_annuncio"])] = x

    for op in ("vendita", "affitto"):
        src = os.path.join(AQUI, "annunci_%s.csv" % op)
        if not os.path.exists(src):
            continue
        with open(src, encoding="utf-8-sig", newline="") as fh:
            rd = csv.DictReader(fh, delimiter=";")
            campi = [c for c in rd.fieldnames if c not in ("spese_mese", "spese_anno", "rif", "categoria_rif", "motivo_rif", "rif_gemini", "categoria_gemini", "motivo_gemini", "rif_grok", "categoria_grok", "motivo_grok", "rif_openai", "categoria_openai", "motivo_openai", "latitudine", "longitudine")] + ["spese_mese", "spese_anno", "rif", "categoria_rif", "motivo_rif", "rif_gemini", "categoria_gemini", "motivo_gemini", "rif_grok", "categoria_grok", "motivo_grok", "rif_openai", "categoria_openai", "motivo_openai", "latitudine", "longitudine"]
            dati = list(rd)
        n = nc = 0
        for r in dati:
            v = righe.get(r["id_annuncio"], (None,))[0]
            r["spese_mese"] = fmt(v)
            r["spese_anno"] = fmt(None if v is None else v * 12)
            x = rif.get((op, r["id_annuncio"]), {})
            r["rif"], r["categoria_rif"], r["motivo_rif"] = x.get("rif", ""), x.get("categoria_rif", ""), x.get("motivo_rif", "")
            g = gem.get((op, r["id_annuncio"]), {})
            r["rif_gemini"], r["categoria_gemini"], r["motivo_gemini"] = g.get("immobile_valido", ""), g.get("categoria_llm", ""), g.get("motivo_llm", "")
            k = grk.get((op, r["id_annuncio"]), {})
            r["rif_grok"], r["categoria_grok"], r["motivo_grok"] = k.get("rif", ""), k.get("categoria", ""), k.get("motivo", "")
            o = oai.get((op, r["id_annuncio"]), {})
            r["rif_openai"], r["categoria_openai"], r["motivo_openai"] = o.get("rif", ""), o.get("categoria", ""), o.get("motivo", "")
            c = coord.get(r["id_annuncio"])
            r["latitudine"], r["longitudine"] = fmt_coord(c, 0), fmt_coord(c, 1)
            n += v is not None
            nc += c is not None
        dst = os.path.join(AQUI, "annunci_%s_spese.csv" % op)
        with open(dst, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campi, delimiter=";", extrasaction="ignore")
            w.writeheader(); w.writerows(dati)
        print("annunci_%s_spese.csv: %d annunci, %d con spese, %d con coordinate" % (op, len(dati), n, nc))
    return 0

if __name__ == "__main__":
    sys.exit(main())
