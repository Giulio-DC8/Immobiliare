# -*- coding: utf-8 -*-
"""
GDC Immobiliare - stima del CANONE DELL'INTERO IMMOBILE per gli annunci "stanza_o_posto_letto".

Input : analisi_modelli/output_affitti.csv  (separatore e codifica riconosciuti e mantenuti)
        si selezionano le righe con categoria_openai = "stanza_o_posto_letto".
Per ognuna si manda a OpenAI la descrizione (titolo, localita', tipologia, mq, locali, canone, descrizione) e si chiede:
  - se nella descrizione e' indicato il canone dell'INTERO immobile  -> si usa quello;
  - altrimenti quante stanze sono affittabili (o l'immobile ha)      -> canone intero = canone della stanza x numero di stanze.
La moltiplicazione e' fatta qui in Python, non da OpenAI. Se non si capisce, il valore resta vuoto.

Output: analisi_modelli/output_affitti_stanze.csv = stesso file + colonne finali
        canone_intero_stimato, metodo_stima, n_stanze_stimate, motivo_stima
        (le righe non "stanza_o_posto_letto" hanno le 4 colonne vuote). Con --sostituisci sovrascrive output_affitti.csv
        (copia prima in _to_delete). Archivio (cache): stanze_intero_cache.csv, un annuncio gia' elaborato non viene rimandato.

Chiave API SOLO da variabile d'ambiente:   set OPENAI_API_KEY=la_tua_chiave

Uso:   python stanze_affitto_intero.py --stima
       python stanze_affitto_intero.py --prova 3
       python stanze_affitto_intero.py
       python stanze_affitto_intero.py --modello gpt-6-luna --sostituisci
Dipendenze: solo libreria standard di Python.
"""
import argparse, csv, io, json, os, shutil, sys, time, urllib.request, urllib.error

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
CATEGORIA = "stanza_o_posto_letto"
CACHE = os.path.join(AQUI, "stanze_intero_cache.csv")
CAMPI_CACHE = ["id_annuncio", "canone", "metodo", "n_stanze", "prezzo_intero", "motivo", "modello"]
NUOVE = ["canone_intero_stimato", "metodo_stima", "n_stanze_stimate", "motivo_stima"]

ISTRUZIONI = (
    "Sei un analista immobiliare. Ti viene dato un annuncio di AFFITTO italiano in cui il canone indicato si riferisce a una "
    "singola stanza o a un posto letto, non all'intero appartamento. Leggi la descrizione e rispondi in JSON:\n"
    "- prezzo_intero_mensile: il canone MENSILE dell'intero immobile in euro, SOLO se e' scritto esplicitamente nel testo "
    "(es. 'intero appartamento 900 euro'); altrimenti null. Non inventare e non sommare spese condominiali.\n"
    "- n_stanze: il numero di stanze/posti letto affittabili dell'immobile, SOLO se ricavabile dal testo "
    "(es. 'appartamento di 3 camere da letto, affittasi stanza singola' = 3; 'bilocale con 2 posti letto' = 2); altrimenti null. "
    "Conta le camere da letto, non i locali di servizio (cucina, bagno, soggiorno).\n"
    "- tipo: 'prezzo_intero' se hai trovato il prezzo dell'intero immobile; 'n_stanze' se hai solo il numero di stanze; "
    "'non_determinabile' se non c'e' nessuna delle due informazioni.\n"
    "- motivo: una frase in italiano con la parte di testo su cui ti basi.\n"
    "Il canone indicato nell'annuncio e' quello della singola stanza/posto letto."
)
SCHEMA = {"type": "object",
          "properties": {"tipo": {"type": "string", "enum": ["prezzo_intero", "n_stanze", "non_determinabile"]},
                         "prezzo_intero_mensile": {"type": ["number", "null"]},
                         "n_stanze": {"type": ["integer", "null"]},
                         "motivo": {"type": "string"}},
          "required": ["tipo", "prezzo_intero_mensile", "n_stanze", "motivo"], "additionalProperties": False}


def leggi_testo(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise RuntimeError("codifica non riconosciuta: " + path)


def testo_annuncio(r):
    campi = [("Titolo", "titolo"), ("Localita", "localita"), ("Tipologia", "tipologia"), ("Metri quadri", "mq"),
             ("Locali", "locali"), ("Canone indicato (EUR/mese, della singola stanza/posto letto)", "prezzo")]
    righe = ["%s: %s" % (a, (r.get(b) or "").strip()) for a, b in campi if (r.get(b) or "").strip()]
    righe.append("Descrizione: " + " ".join((r.get("descrizione") or "").split())[:4000])
    return "\n".join(righe)


def chiama(modello, chiave, testo, tentativi=8):
    corpo = {"model": modello,
             "messages": [{"role": "system", "content": ISTRUZIONI}, {"role": "user", "content": testo}],
             "response_format": {"type": "json_schema", "json_schema": {"name": "stanze", "schema": SCHEMA, "strict": True}}}
    dati = json.dumps(corpo).encode("utf-8")
    attesa = 15
    for t in range(tentativi):
        req = urllib.request.Request(BASE + "/chat/completions", data=dati, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + chiave,
                                              "User-Agent": "gdc-immobiliare/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                j = json.loads(resp.read().decode("utf-8"))
            contenuto = ((j.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            if not contenuto.strip():
                raise ValueError("risposta vuota")
            return json.loads(contenuto)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            if e.code in (400, 401, 403, 404):
                raise SystemExit("Errore OpenAI %d (non ritentabile): %s\nControlla OPENAI_API_KEY e --modello." % (e.code, msg))
            if e.code == 429 and "insufficient_quota" in msg:
                raise SystemExit("Errore OpenAI 429: credito esaurito (insufficient_quota).\n" + msg)
            try:
                ra = int(e.headers.get("Retry-After", "0"))
            except (TypeError, ValueError):
                ra = 0
            p = max(attesa, ra)
            print("   HTTP %d, tentativo %d/%d, nuovo tentativo tra %ds" % (e.code, t + 1, tentativi, p))
            time.sleep(p); attesa = min(attesa * 2, 180)
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            print("   risposta non valida (%s: %s), tentativo %d/%d" % (type(e).__name__, str(e)[:100], t + 1, tentativi))
            time.sleep(attesa); attesa = min(attesa * 2, 180)
    return None


def numero(v):
    try:
        return float(str(v).strip().replace(".", "").replace(",", ".")) if "," in str(v) else float(str(v).strip())
    except ValueError:
        return None


def risolvi(canone, esito):
    """-> (canone_intero, metodo, n_stanze, prezzo_intero_dal_testo)"""
    pi, n = esito.get("prezzo_intero_mensile"), esito.get("n_stanze")
    if esito.get("tipo") == "prezzo_intero" and pi and pi > 0:
        return round(pi), "prezzo_intero_in_descrizione", n, pi
    if n and n >= 1 and canone:
        return round(canone * n), "canone_stanza_x_n_stanze", n, pi
    return None, "non_determinabile", n, pi


def carica_cache():
    c = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh, delimiter=";"):
                c[r["id_annuncio"]] = r
    return c


def salva_cache(c):
    tmp = CACHE + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPI_CACHE, delimiter=";")
        w.writeheader(); w.writerows(c.values())
    os.replace(tmp, CACHE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=os.path.join("analisi_modelli", "output_affitti.csv"))
    ap.add_argument("--modello", default="gpt-6-luna")
    ap.add_argument("--pausa", type=float, default=0.0, help="secondi tra una chiamata e l'altra (default 0)")
    ap.add_argument("--prova", type=int, help="elabora solo N annunci")
    ap.add_argument("--stima", action="store_true", help="conta soltanto, senza chiamare OpenAI")
    ap.add_argument("--rifai", action="store_true", help="ignora l'archivio e rielabora tutto")
    ap.add_argument("--sostituisci", action="store_true", help="sovrascrive output_affitti.csv (copia in _to_delete)")
    a = ap.parse_args()
    path = a.file if os.path.isabs(a.file) else os.path.join(AQUI, a.file)
    if not os.path.isfile(path):
        print("File non trovato:", path); return 1
    testo, enc = leggi_testo(path)
    prima = testo.splitlines()[0]
    sep = ";" if prima.count(";") > prima.count(",") else ","
    righe = list(csv.DictReader(io.StringIO(testo, newline=""), delimiter=sep))
    campi = [c.lstrip("﻿") for c in list(righe[0].keys())]
    for r in righe:
        if "﻿" + campi[0] in r:
            r[campi[0]] = r.pop("﻿" + campi[0])
    for c in ("categoria_openai", "descrizione", "id_annuncio"):
        if c not in campi:
            print("Colonna mancante:", c); return 1
    campi = [c for c in campi if c not in NUOVE]
    sel = [r for r in righe if (r.get("categoria_openai") or "").strip() == CATEGORIA]
    cache = carica_cache()
    lavoro = [r for r in sel if a.rifai or r["id_annuncio"] not in cache]
    print("Annunci totali: %d | '%s': %d | da elaborare: %d | gia' in archivio: %d" % (
        len(righe), CATEGORIA, len(sel), len(lavoro), len(sel) - len(lavoro)))
    if a.stima:
        print("Chiamate previste a OpenAI:", len(lavoro)); return 0
    chiave = os.environ.get("OPENAI_API_KEY", "").strip()
    if lavoro and not chiave:
        print("Manca la chiave. In Anaconda Prompt:  set OPENAI_API_KEY=la_tua_chiave"); return 1
    if a.prova:
        lavoro = lavoro[:a.prova]
    for k, r in enumerate(lavoro, 1):
        esito = chiama(a.modello, chiave, testo_annuncio(r))
        if esito is None:
            print("[%d/%d] %s senza risposta, salto (riprova al prossimo lancio)" % (k, len(lavoro), r["id_annuncio"])); continue
        canone = numero(r.get("prezzo"))
        tot, metodo, n, pi = risolvi(canone, esito)
        cache[r["id_annuncio"]] = dict(id_annuncio=r["id_annuncio"], canone=r.get("prezzo", ""), metodo=metodo,
                                       n_stanze="" if n is None else n, prezzo_intero="" if pi is None else pi,
                                       motivo=" ".join(str(esito.get("motivo", "")).split()), modello=a.modello)
        salva_cache(cache)
        print("[%d/%d] %s | canone stanza %s | %s | n_stanze %s | intero %s" % (
            k, len(lavoro), r["id_annuncio"], r.get("prezzo", ""), metodo, n if n is not None else "-", tot if tot else "-"))
        if a.pausa:
            time.sleep(a.pausa)
    n_ok = n_no = n_mancano = 0
    out = []
    for r in righe:
        c = cache.get(r["id_annuncio"]) if (r.get("categoria_openai") or "").strip() == CATEGORIA else None
        nuovo = ["", "", "", ""]
        if c:
            canone = numero(c["canone"])
            esito = {"tipo": "prezzo_intero" if c["metodo"] == "prezzo_intero_in_descrizione" else
                     ("n_stanze" if c["n_stanze"] != "" else "non_determinabile"),
                     "prezzo_intero_mensile": numero(c["prezzo_intero"]) if c["prezzo_intero"] != "" else None,
                     "n_stanze": int(float(c["n_stanze"])) if c["n_stanze"] != "" else None}
            tot, metodo, n, _ = risolvi(canone, esito)
            nuovo = [str(tot) if tot else "", metodo, "" if n is None else str(n), c["motivo"]]
            n_ok += bool(tot); n_no += not tot
        elif (r.get("categoria_openai") or "").strip() == CATEGORIA:
            n_mancano += 1
        out.append([r.get(k, "") for k in campi] + nuovo)
    dest = path if a.sostituisci else path.replace(".csv", "_stanze.csv")
    if a.sostituisci:
        os.makedirs(os.path.join(AQUI, "_to_delete"), exist_ok=True)
        shutil.copy2(path, os.path.join(AQUI, "_to_delete", os.path.basename(path).replace(".csv", "_prima_stanze.csv")))
    with open(dest, "w", encoding=enc, newline="", errors="replace") as fh:
        w = csv.writer(fh, delimiter=sep, lineterminator="\r\n")
        w.writerow(campi + NUOVE); w.writerows(out)
    print("Scritto %s | stanze con canone intero stimato: %d | non determinabili: %d | non ancora elaborate: %d" % (
        os.path.basename(dest), n_ok, n_no, n_mancano))
    return 0


if __name__ == "__main__":
    sys.exit(main())
