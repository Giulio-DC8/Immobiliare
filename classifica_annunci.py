# -*- coding: utf-8 -*-
"""Classifica con Gemini (API) se il prezzo di ogni annuncio si riferisce a un IMMOBILE vero e proprio
oppure ad altro (garage, posto auto, multiproprieta' a settimane, affitto stagionale, nuda proprieta', ecc.).

Per ogni annuncio invia a Gemini titolo, tipologia, localita', prezzo, mq, locali, piano e descrizione.
Chiave API SOLO da variabile d'ambiente:   set GEMINI_API_KEY=la_tua_chiave      (Anaconda Prompt / Windows)

Input/Output: annunci_vendita_spese.csv / annunci_affitto_spese.csv (lo STESSO file: aggiunge/aggiorna le colonne
        rif_gemini (SI | NO), categoria_gemini, motivo_gemini; le altre colonne, compresa rif, restano invariate).
        rif_gemini = SI solo se Gemini risponde "immobile"; "incerto" e tutte le altre categorie = NO
        (la categoria_gemini permette di distinguere gli "incerto").
        classificazioni_llm.csv = archivio (cache): un annuncio gia' classificato non viene rimandato a Gemini.
        Prima della scrittura il file originale viene copiato in _to_delete/ (<nome>_prima_gemini.csv).

Esempi:
  python classifica_annunci.py --stima                    (conta le chiamate, non chiama Gemini)
  python classifica_annunci.py --prova 5                  (solo 5 annunci)
  python classifica_annunci.py --operazione vendita
  python classifica_annunci.py                            (vendita + affitto; riprende da dove era arrivato)
  python classifica_annunci.py --modello gemini-3.8-flash --pausa 1.5
"""
import argparse, csv, hashlib, json, os, shutil, sys, time, urllib.request, urllib.error

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
CACHE = os.path.join(AQUI, "classificazioni_llm.csv")
CAMPI_CACHE = ["id_annuncio", "operazione", "hash", "categoria_llm", "immobile_valido", "motivo_llm", "modello"]

NUOVE = ["rif_gemini", "categoria_gemini", "motivo_gemini"]
CATEGORIE = ["immobile", "garage_box_posto_auto", "multiproprieta_settimana", "affitto_stagionale_turistico",
             "nuda_proprieta_o_quota", "stanza_o_posto_letto", "terreno_o_rudere", "altro", "incerto"]

ISTRUZIONI = """Sei un analista di annunci immobiliari italiani. Ricevi i dati di UN annuncio di {op_txt}.
Devi stabilire a COSA si riferisce il prezzo indicato, leggendo titolo, tipologia e soprattutto la descrizione.

Categorie (rispondi con una sola):
- immobile: il prezzo riguarda un'intera unita' abitativa (appartamento, villetta, casa, attico, loft, ecc.) utilizzabile come abitazione, {extra_ok}
- garage_box_posto_auto: il prezzo riguarda in realta' un garage, box, cantina, posto auto o simili (anche se il titolo dice appartamento).
- multiproprieta_settimana: quota a settimane / multiproprieta' / uso a periodi.
- affitto_stagionale_turistico: {extra_stag}
- nuda_proprieta_o_quota: nuda proprieta', usufrutto, quota di proprieta' parziale, diritto di abitazione, vitalizio.
- stanza_o_posto_letto: singola stanza o posto letto in un appartamento condiviso.
- terreno_o_rudere: terreno, rustico da ristrutturare, rudere, immobile non abitabile o in costruzione sulla carta.
- altro: negozio, ufficio, magazzino, o qualsiasi altro oggetto non abitativo.
- incerto: la descrizione non permette di capire a cosa si riferisca il prezzo.

Regole: usa 'immobile' solo se il prezzo si riferisce chiaramente all'intera abitazione. Se la descrizione parla di un oggetto diverso da quello del titolo (es. titolo 'bilocale' ma testo su un garage), scegli la categoria dell'oggetto descritto. Non inventare: se i dati sono contraddittori o insufficienti scegli 'incerto'.
Rispondi in JSON con: categoria, motivo (massimo 20 parole, in italiano)."""

ARGOMENTI = {
    "vendita": dict(op_txt="VENDITA",
                    extra_ok="venduta per intero (anche se occupata o da ristrutturare ma comunque abitazione).",
                    extra_stag="uso turistico/stagionale/periodi limitati proposto in vendita o affitto non a lungo termine."),
    "affitto": dict(op_txt="AFFITTO",
                    extra_ok="affittata per intero con canone mensile a lungo termine o transitorio.",
                    extra_stag="affitto turistico, per settimane, mesi estivi o brevi periodi, canone non mensile ordinario."),
}

SCHEMA = {"type": "OBJECT",
          "properties": {"categoria": {"type": "STRING", "enum": CATEGORIE}, "motivo": {"type": "STRING"}},
          "required": ["categoria", "motivo"]}

import lotto as L
SCHEMA_LOTTO = L.schema_lotto(CATEGORIE, True)


def testo_annuncio(r):
    d = (r.get("descrizione") or "").strip()
    if len(d) > 3000:
        d = d[:3000] + " [...]"
    return ("Titolo: %s\nTipologia: %s\nLocalita': %s\nPrezzo indicato: %s euro\nSuperficie: %s mq\nLocali: %s\nPiano: %s\nDescrizione: %s"
            % (r.get("titolo", ""), r.get("tipologia", ""), r.get("localita", ""), r.get("prezzo", ""),
               r.get("mq", ""), r.get("locali", ""), r.get("piano", ""), d or "(assente)"))


def estrai_testo(j):
    """Estrae il testo JSON generato, anche se la risposta contiene piu' parti (es. parti 'thought')
    o se il modello non ha prodotto candidati (blocco/filtri)."""
    cands = j.get("candidates") or []
    if not cands:
        raise ValueError("nessun candidato nella risposta: %s" % json.dumps(j.get("promptFeedback", j))[:200])
    parti = (cands[0].get("content") or {}).get("parts") or []
    testo = "".join(p.get("text", "") for p in parti if isinstance(p, dict) and not p.get("thought"))
    if not testo.strip():
        raise ValueError("risposta senza testo (finishReason=%s)" % cands[0].get("finishReason"))
    return testo


def chiama(modello, chiave, op, testo, tentativi=8, lotto=False):
    url = "%s/models/%s:generateContent" % (BASE, modello)
    corpo = {"systemInstruction": {"parts": [{"text": ISTRUZIONI.format(**ARGOMENTI[op]) + (L.SUFFISSO if lotto else "")}]},
             "contents": [{"role": "user", "parts": [{"text": testo}]}],
             "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": SCHEMA_LOTTO if lotto else SCHEMA}}
    dati = json.dumps(corpo).encode("utf-8")
    attesa = 15
    for t in range(tentativi):
        req = urllib.request.Request(url, data=dati, method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": chiave})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                j = json.loads(resp.read().decode("utf-8"))
            grezzo = estrai_testo(j)
            if lotto:
                return grezzo
            esito = json.loads(grezzo)
            cat = esito.get("categoria", "incerto")
            if cat not in CATEGORIE:
                cat = "incerto"
            return cat, " ".join(str(esito.get("motivo", "")).split())
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            if e.code in (400, 401, 403, 404):
                raise SystemExit("Errore Gemini %d (non ritentabile): %s\n"
                                 "Controlla chiave (GEMINI_API_KEY) e nome del modello (--modello)." % (e.code, msg))
            try:
                ra = int(e.headers.get("Retry-After", "0"))
            except (TypeError, ValueError):
                ra = 0
            pausa = max(attesa, ra)
            print("   HTTP %d (server Gemini sovraccarico/limite), tentativo %d/%d, nuovo tentativo tra %ds"
                  % (e.code, t + 1, tentativi, pausa))
            time.sleep(pausa); attesa = min(attesa * 2, 180)
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            print("   risposta non valida (%s: %s), tentativo %d/%d, nuovo tentativo tra %ds"
                  % (type(e).__name__, str(e)[:120], t + 1, tentativi, attesa))
            time.sleep(attesa); attesa = min(attesa * 2, 180)
    return None


def carica_cache():
    c = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh, delimiter=";"):
                c[(r["operazione"], r["id_annuncio"])] = r
    return c


def salva_cache(c):
    tmp = CACHE + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPI_CACHE, delimiter=";")
        w.writeheader()
        for r in c.values():
            w.writerow(r)
    os.replace(tmp, CACHE)


def valido(cat):
    return "SI" if cat == "immobile" else "NO"


def sorgente(op):
    f = os.path.join(AQUI, "annunci_%s_spese.csv" % op)
    return f if os.path.exists(f) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--operazione", choices=["vendita", "affitto", "entrambe"], default="entrambe")
    p.add_argument("--modello", default="gemini-3.8-flash")
    p.add_argument("--pausa", type=float, default=1.0, help="secondi tra una chiamata e l'altra")
    p.add_argument("--batch", type=int, default=30, help="annunci per chiamata (1 = uno alla volta)")
    p.add_argument("--prova", type=int, help="elabora solo N annunci (da classificare)")
    p.add_argument("--stima", action="store_true", help="conta soltanto, senza chiamare Gemini")
    p.add_argument("--rifai", action="store_true", help="ignora l'archivio e riclassifica tutto")
    a = p.parse_args()

    ops = ["vendita", "affitto"] if a.operazione == "entrambe" else [a.operazione]
    chiave = os.environ.get("GEMINI_API_KEY", "").strip()
    if not chiave and not a.stima:
        print("Manca la chiave. In Anaconda Prompt:  set GEMINI_API_KEY=la_tua_chiave"); return 1

    cache = carica_cache()
    lavoro, tabelle = [], {}
    for op in ops:
        f = sorgente(op)
        if not f:
            print("Nessun file annunci per", op); continue
        with open(f, encoding="utf-8-sig", newline="") as fh:
            rd = csv.DictReader(fh, delimiter=";")
            campi = [c for c in rd.fieldnames if c not in NUOVE]
            righe = list(rd)
        tabelle[op] = (campi, righe)
        for r in righe:
            h = hashlib.md5(testo_annuncio(r).encode("utf-8")).hexdigest()
            c = cache.get((op, r["id_annuncio"]))
            if a.rifai or not c or c["hash"] != h:
                lavoro.append((op, r, h))
    print("Annunci totali: %d | da classificare: %d | gia' in archivio: %d"
          % (sum(len(t[1]) for t in tabelle.values()), len(lavoro), sum(len(t[1]) for t in tabelle.values()) - len(lavoro)))
    if a.stima:
        print("Chiamate previste a Gemini:", len(lavoro)); return 0
    if a.prova:
        lavoro = lavoro[:a.prova]

    def reg(op, r, h, cat, motivo):
        cache[(op, r["id_annuncio"])] = dict(id_annuncio=r["id_annuncio"], operazione=op, hash=h, categoria_llm=cat,
                                             immobile_valido=valido(cat), motivo_llm=motivo, modello=a.modello)
    L.elabora(lavoro, max(1, a.batch), a.pausa,
              lambda op, r: chiama(a.modello, chiave, op, testo_annuncio(r)),
              lambda op, voci: chiama(a.modello, chiave, op, L.testo_lotto([(i, testo_annuncio(r)) for i, r in voci]), lotto=True),
              reg, lambda: salva_cache(cache), CATEGORIE)
    salva_cache(cache)

    for op, (campi, righe) in tabelle.items():
        n = {"SI": 0, "NO": 0, "": 0}
        for r in righe:
            c = cache.get((op, r["id_annuncio"]))
            r["rif_gemini"] = c["immobile_valido"] if c else ""
            r["categoria_gemini"] = c["categoria_llm"] if c else ""
            r["motivo_gemini"] = c["motivo_llm"] if c else ""
            n[r["rif_gemini"]] += 1
        dst = os.path.join(AQUI, "annunci_%s_spese.csv" % op)
        bk = os.path.join(AQUI, "_to_delete")
        os.makedirs(bk, exist_ok=True)
        shutil.copyfile(dst, os.path.join(bk, "annunci_%s_spese_prima_gemini.csv" % op))
        tmp = dst + ".tmp"
        with open(tmp, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campi + NUOVE, delimiter=";")
            w.writeheader(); w.writerows(righe)
        os.replace(tmp, dst)
        confr = sum(1 for r in righe if r.get("rif") and r["rif_gemini"] and r["rif"] != r["rif_gemini"])
        print("%s: rif_gemini SI %d | NO %d | non ancora classificati %d | discordanti da rif: %d  ->  %s"
              % (op, n["SI"], n["NO"], n[""], confr, os.path.basename(dst)))
    return 0

if __name__ == "__main__":
    sys.exit(main())
