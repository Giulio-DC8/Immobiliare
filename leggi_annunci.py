#!/usr/bin/env python3
"""
leggi_annunci.py - estrae gli annunci dalle pagine Idealista (vista ELENCO) salvate a mano
in dati\\pagine\\  (Ctrl+S -> "Pagina web, solo HTML"). Lavora SOLO su file locali: nessuna richiesta a Idealista.

Cosa fa
  - legge tutti i file .html/.htm della cartella (default: dati\\pagine);
  - per ogni annuncio estrae: id, titolo, tipologia, localita', prezzo, prezzo precedente e ribasso %,
    locali, mq, piano, ascensore, box, descrizione, link, inserzionista professionale/privato;
  - dal canonical della pagina ricava l'operazione (vendita/affitto) e la forma disegnata (shape):
    il centro della forma viene abbinato all'area piu' vicina di aree_vendita.csv / aree_affitto.csv
    (entro --tolleranza metri) per riportare id area, citta' e poli;
  - calcola prezzo al mq (vendita) o canone al mq (affitto);
  - dalle SCHEDE DI DETTAGLIO salvate in dati\\pagine\\pagina_ID (file ID.html o pagina_ID.html; vedi --schede) aggiunge, per
    ogni annuncio di cui c'e' la scheda: spese_mese e spese_anno (spese condominiali) e latitudine e longitudine (centro della
    mappa nella scheda, che puo' essere esatto o approssimato). Se la scheda manca le colonne restano vuote
    (o conservano il valore del lancio precedente). Usa le funzioni di leggi_spese.py (stessa cartella);
  - deduplica per id annuncio (le aree si sovrappongono) e AGGIORNA annunci.csv senza perdere la prima
    rilevazione e il primo prezzo visto (per individuare ribassi nel tempo).

Uso
    python leggi_annunci.py
    python leggi_annunci.py --cartella dati\\pagine
    python leggi_annunci.py --schede dati\\pagine\\pagina_ID     # cartella delle schede di dettaglio (default)
    python leggi_annunci.py --nuovo        # ignora i file annunci_*.csv esistenti e riparte da zero

Nota: i campi si leggono dalla struttura della pagina attuale; se Idealista la cambia, lo script lo
segnala (annunci trovati ma senza prezzo) invece di produrre dati vuoti in silenzio.
"""
import argparse, csv, datetime, glob, html, io, math, os, re, sys, urllib.parse

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from leggi_spese import leggi as leggi_scheda, analizza as analizza_spese, coordinate as coordinate_scheda, fmt as fmt_spese, fmt_coord
SITO = "https://www.idealista.it"
COLONNE = ["id_annuncio", "operazione", "tipologia", "titolo", "localita", "prezzo", "prezzo_precedente",
           "ribasso_pct", "mq", "prezzo_mq", "locali", "piano", "ascensore", "box", "professionale",
           "descrizione", "link", "id_area", "citta_area", "poli_area", "file_sorgente",
           "prima_rilevazione", "ultima_rilevazione", "prezzo_prima_rilevazione", "presente_ultimo_lancio",
           "spese_mese", "spese_anno", "latitudine", "longitudine"]


# ---------- polyline (stesso formato di genera_aree.py) ----------
def decode_polyline(e):
    i = lat = lng = 0
    pts = []
    while i < len(e):
        for k in (0, 1):
            shift = res = 0
            while True:
                b = ord(e[i]) - 63
                i += 1
                res |= (b & 31) << shift
                shift += 5
                if b < 32:
                    break
            d = ~(res >> 1) if res & 1 else res >> 1
            if k == 0:
                lat += d
            else:
                lng += d
        pts.append((lat / 1e5, lng / 1e5))
    return pts


def hav(a, b):
    R = 6371000.0
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def leggi_testo(p):
    raw = open(p, "rb").read()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def pulisci(t):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t or ""))).strip()


def numero(t):
    t = re.sub(r"[^\d]", "", t or "")
    return int(t) if t else ""


# ---------- aree ----------
def carica_aree(base):
    aree = []
    for nome in ("aree_vendita.csv", "aree_affitto.csv"):
        p = os.path.join(base, nome)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(io.StringIO(leggi_testo(p)), delimiter=";"):
            aree.append({"op": r["operazione"], "id": r["id_area"], "citta": r["citta"],
                         "poli": r["enti_inclusi"], "c": (float(r["lat"]), float(r["lon"]))})
    return aree


def info_pagina(s):
    """(operazione, centro della shape) dal canonical della pagina."""
    m = re.search(r'rel="canonical" href="([^"]+)"', s)
    if not m:
        return "", None
    url = html.unescape(m.group(1))
    op = "vendita" if "/vendita-" in url else ("affitto" if "/affitto-" in url else "")
    sh = re.search(r"shape=([^&]+)", url)
    if not sh:
        return op, None
    forma = urllib.parse.unquote(sh.group(1)).strip("()")
    try:
        pts = decode_polyline(forma)[:-1] or decode_polyline(forma)
        return op, (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    except Exception:
        return op, None


def abbina_area(op, centro, aree, tol):
    if not centro:
        return None
    migliore = None
    for a in aree:
        if op and a["op"] != op:
            continue
        d = hav(centro, a["c"])
        if d <= tol and (migliore is None or d < migliore[0]):
            migliore = (d, a)
    return migliore[1] if migliore else None


# ---------- annunci ----------
def estrai_annunci(s):
    out = []
    for a in re.findall(r"<article\b.*?</article>", s, re.S):
        mid = re.search(r'<article[^>]*class="[^"]*\bitem\b[^"]*"[^>]*data-element-id="(\d+)"', a)
        if not mid:
            continue
        titolo = re.search(r'class="item-link[^"]*"[^>]*title="([^"]*)"', a)
        titolo = html.unescape(titolo.group(1)).strip() if titolo else ""
        prezzo = re.search(r'class="item-price[^"]*">\s*([\d.]+)', a)
        prec = re.search(r'pricedown_price">\s*([\d.]+)', a)
        ribasso = re.search(r'pricedown_icon[^>]*>\s*([\d.,]+)\s*%', a)
        det = [pulisci(x) for x in re.findall(r'class="item-detail">(.*?)</span>', a, re.S)]
        desc = re.search(r'class="item-description[^"]*"[^>]*>(.*?)</div>', a, re.S)
        locali = mq = piano = ""
        box = "no"
        for d in det:
            if re.match(r"^\d+\s*local", d, re.I):
                locali = numero(d)
            elif re.search(r"m²|mq", d):
                mq = numero(d)
            elif re.search(r"piano|attico|terra|rialzato|seminterrato", d, re.I):
                piano = d
            elif re.match(r"^box", d, re.I):
                box = "sì"
        tip, loc = "", titolo
        mt = re.match(r"^(.+?)\s+(?:in|a)\s+(.*)$", titolo)
        if mt:
            tip, loc = mt.group(1), mt.group(2)
        out.append({
            "id_annuncio": mid.group(1), "titolo": titolo, "tipologia": tip, "localita": loc,
            "prezzo": numero(prezzo.group(1)) if prezzo else "",
            "prezzo_precedente": numero(prec.group(1)) if prec else "",
            "ribasso_pct": ribasso.group(1) if ribasso else "",
            "mq": mq, "locali": locali, "piano": piano,
            "ascensore": ("con" if re.search(r"con ascensore", piano, re.I) else
                          ("senza" if re.search(r"senza ascensore", piano, re.I) else "")),
            "box": box,
            "professionale": "sì" if 'data-is-professional-ad="true"' in a else "no",
            "descrizione": pulisci(desc.group(1))[:300] if desc else "",
            "link": "%s/immobile/%s/" % (SITO, mid.group(1)),
        })
    return out


def carica_schede(cart):
    """{id annuncio: percorso della scheda di dettaglio} (file ID.ext oppure pagina_ID.ext)."""
    ind = {}
    if not os.path.isdir(cart):
        return ind
    for nome in sorted(os.listdir(cart)):
        m = re.match(r"^(?:pagina_)?(\d+)(\.[A-Za-z0-9]+)?$", nome, re.I)
        p = os.path.join(cart, nome)
        if m and os.path.isfile(p):
            ind.setdefault(m.group(1), p)
    return ind


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartella", default=os.path.join("dati", "pagine"))
    ap.add_argument("--out", default=None, help="file unico; se omesso: annunci_vendita.csv e annunci_affitto.csv")
    ap.add_argument("--schede", default=os.path.join("dati", "pagine", "pagina_ID"), help="cartella delle schede di dettaglio (spese e coordinate)")
    ap.add_argument("--tolleranza", type=int, default=150, help="metri per abbinare la forma disegnata a un'area")
    ap.add_argument("--nuovo", action="store_true", help="ignora l'annunci.csv esistente")
    a = ap.parse_args(argv)
    P = lambda x: x if os.path.isabs(x) else os.path.join(BASE, x)

    files = sorted(f for f in glob.glob(os.path.join(P(a.cartella), "*")) if f.lower().endswith((".html", ".htm", ".txt")))
    if not files:
        sys.exit("Nessun file .html o .txt in %s" % P(a.cartella))
    aree = carica_aree(BASE)

    destinazioni = [a.out] if a.out else ["annunci_vendita.csv", "annunci_affitto.csv"]
    precedenti = {}
    if not a.nuovo:
        for d in destinazioni:
            if os.path.exists(P(d)):
                for r in csv.DictReader(io.StringIO(leggi_testo(P(d))), delimiter=";"):
                    precedenti[r["id_annuncio"]] = r

    trovati, ignorate, senza_prezzo, non_abbinate = {}, [], 0, []
    for f in files:
        s = leggi_testo(f)
        nome = os.path.basename(f)
        ann = estrai_annunci(s)
        if not ann:
            ignorate.append(nome)
            continue
        op, centro = info_pagina(s)
        area = abbina_area(op, centro, aree, a.tolleranza)
        if not area:
            non_abbinate.append(nome)
        data = datetime.date.fromtimestamp(os.path.getmtime(f)).isoformat()
        for r in ann:
            if r["prezzo"] == "":
                senza_prezzo += 1
            r.update(operazione=op, file_sorgente=nome, ultima_rilevazione=data,
                     id_area=area["id"] if area else "", citta_area=area["citta"] if area else "",
                     poli_area=area["poli"] if area else "")
            r["prezzo_mq"] = (round(r["prezzo"] / r["mq"]) if r["prezzo"] != "" and r["mq"] not in ("", 0) else "")
            if r["id_annuncio"] in trovati:   # stesso annuncio in piu' aree/pagine: unisce le aree
                t = trovati[r["id_annuncio"]]
                for k_src, k_dst in (("id_area", "id_area"), ("citta_area", "citta_area")):
                    if r[k_src] and r[k_src] not in t[k_dst].split("|"):
                        t[k_dst] = "|".join(x for x in (t[k_dst], r[k_src]) if x)
                continue
            trovati[r["id_annuncio"]] = r

    for i, r in trovati.items():
        p = precedenti.get(i)
        r["prima_rilevazione"] = (p or {}).get("prima_rilevazione") or r["ultima_rilevazione"]
        r["prezzo_prima_rilevazione"] = (p or {}).get("prezzo_prima_rilevazione") or r["prezzo"]
        r["presente_ultimo_lancio"] = "sì"

    # annunci gia' nei file ma non presenti nelle pagine di questo lancio: si conservano (non si perde lo storico)
    for i, p in precedenti.items():
        if i not in trovati:
            p = dict(p)
            p["presente_ultimo_lancio"] = "no"
            trovati[i] = p
    # spese condominiali e coordinate dalle schede di dettaglio (se salvate)
    ind = carica_schede(P(a.schede))
    n_sched = n_spese = n_coord = 0
    for i, r in trovati.items():
        if i in ind:
            n_sched += 1
            try:
                testo = leggi_scheda(ind[i])
                v, _stato = analizza_spese(testo, i)
                c = coordinate_scheda(testo)
            except Exception:
                v, c = None, None
            r["spese_mese"], r["spese_anno"] = fmt_spese(v), fmt_spese(None if v is None else v * 12)
            r["latitudine"], r["longitudine"] = fmt_coord(c, 0), fmt_coord(c, 1)
        else:   # scheda non disponibile: si conservano i valori del lancio precedente (se c'erano)
            p0 = precedenti.get(i) or {}
            for k in ("spese_mese", "spese_anno", "latitudine", "longitudine"):
                r[k] = r.get(k) or p0.get(k, "")
        n_spese += str(r.get("spese_mese", "")) != ""
        n_coord += str(r.get("latitudine", "")) != ""
    righe = sorted(trovati.values(), key=lambda r: (r["citta_area"], r["id_area"], float(r["prezzo"]) if str(r["prezzo"]).strip() != "" else 0.0))
    if a.out:
        gruppi = {a.out: righe}
    else:
        gruppi = {"annunci_vendita.csv": [r for r in righe if r["operazione"] != "affitto"],
                  "annunci_affitto.csv": [r for r in righe if r["operazione"] == "affitto"]}
    for nome_out, rr in gruppi.items():
        if not a.out and not rr and not os.path.exists(P(nome_out)):
            continue            # nessun annuncio di questo tipo: non crea file vuoti
        with open(P(nome_out), "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLONNE, delimiter=";", extrasaction="ignore")
            w.writeheader()
            w.writerows(rr)
        print("Scritti %d annunci in %s" % (len(rr), nome_out))

    print("File letti: %d | annunci unici: %d" % (len(files), len(righe)))
    if not ind:
        print("ATTENZIONE: nessuna scheda di dettaglio trovata in %s: spese e coordinate restano vuote." % P(a.schede))
    else:
        print("Schede di dettaglio trovate: %d | annunci con scheda: %d | con spese: %d | con coordinate: %d (su %d annunci)"
              % (len(ind), n_sched, n_spese, n_coord, len(righe)))
    nuovi = [i for i in trovati if i not in precedenti]
    if precedenti:
        print("Nuovi rispetto alla volta precedente: %d | gia' presenti: %d | non piu' visti: %d"
              % (len(nuovi), len(trovati) - len(nuovi), len([i for i in precedenti if i not in trovati])))
    if ignorate:
        print("ATTENZIONE: nessun annuncio trovato in: %s\n  -> pagina in vista mappa o elenco vuoto? Salvare la vista ELENCO." % ", ".join(ignorate))
    if senza_prezzo:
        print("ATTENZIONE: %d annunci senza prezzo: la struttura della pagina potrebbe essere cambiata." % senza_prezzo)
    if non_abbinate:
        print("Nota: forma disegnata non abbinata a un'area di aree_*.csv (id_area vuoto) in: %s" % ", ".join(non_abbinate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
