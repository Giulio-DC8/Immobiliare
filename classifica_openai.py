# -*- coding: utf-8 -*-
"""Classifica con ChatGPT (API OpenAI) se il prezzo di ogni annuncio si riferisce a un IMMOBILE vero e proprio.
Usa lo stesso prompt e le stesse categorie di classifica_annunci.py (Gemini) e classifica_openai.py.

Chiave API SOLO da variabile d'ambiente:   set OPENAI_API_KEY=la_tua_chiave      (Anaconda Prompt / Windows)

Input/Output: annunci_vendita_spese.csv / annunci_affitto_spese.csv (STESSO file: aggiunge/aggiorna
        rif_openai (SI | NO), categoria_openai, motivo_openai; le altre colonne restano invariate).
        rif_openai = SI solo se ChatGPT risponde "immobile"; "incerto" e le altre categorie = NO.
        classificazioni_openai.csv = archivio (cache). Backup in _to_delete/ prima della scrittura.

Esempi:
  python classifica_openai.py --stima
  python classifica_openai.py --prova 5
  python classifica_openai.py --operazione vendita
  python classifica_openai.py --modello gpt-6-luna --pausa 6
"""
import argparse, csv, hashlib, json, os, shutil, sys, time, urllib.request, urllib.error

from classifica_annunci import ISTRUZIONI, ARGOMENTI, CATEGORIE, testo_annuncio

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
CACHE = os.path.join(AQUI, "classificazioni_openai.csv")
CAMPI_CACHE = ["id_annuncio", "operazione", "hash", "categoria", "rif", "motivo", "modello"]
NUOVE = ["rif_openai", "categoria_openai", "motivo_openai"]

SCHEMA = {"type": "object",
          "properties": {"categoria": {"type": "string", "enum": CATEGORIE}, "motivo": {"type": "string"}},
          "required": ["categoria", "motivo"], "additionalProperties": False}

import lotto as L
SCHEMA_LOTTO = L.schema_lotto(CATEGORIE, False)


def chiama(modello, chiave, op, testo, tentativi=8, lotto=False):
    corpo = {"model": modello,
             "messages": [{"role": "system", "content": ISTRUZIONI.format(**ARGOMENTI[op]) + (L.SUFFISSO if lotto else "")},
                          {"role": "user", "content": testo}],
             "response_format": {"type": "json_schema",
                                 "json_schema": {"name": "classificazione", "schema": SCHEMA_LOTTO if lotto else SCHEMA, "strict": True}}}
    dati = json.dumps(corpo).encode("utf-8")
    attesa = 15
    for t in range(tentativi):
        req = urllib.request.Request(BASE + "/chat/completions", data=dati, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + chiave,
                                              "User-Agent": "gdc-immobiliare/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                j = json.loads(resp.read().decode("utf-8"))
            scelte = j.get("choices") or []
            if not scelte:
                raise ValueError("nessuna scelta nella risposta: %s" % json.dumps(j)[:200])
            contenuto = (scelte[0].get("message") or {}).get("content") or ""
            if not contenuto.strip():
                raise ValueError("risposta vuota (finish_reason=%s)" % scelte[0].get("finish_reason"))
            if lotto:
                return contenuto
            esito = json.loads(contenuto)
            cat = esito.get("categoria", "incerto")
            if cat not in CATEGORIE:
                cat = "incerto"
            return cat, " ".join(str(esito.get("motivo", "")).split())
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            if e.code in (400, 401, 403, 404):
                raise SystemExit("Errore OpenAI %d (non ritentabile): %s\n"
                                 "Controlla chiave (OPENAI_API_KEY) e nome del modello (--modello)." % (e.code, msg))
            if e.code == 429 and "insufficient_quota" in msg:
                raise SystemExit("Errore OpenAI 429: credito/quota esaurita (insufficient_quota). "
                                 "Controlla piano e credito API su platform.openai.com (Billing).\n" + msg)
            try:
                ra = int(e.headers.get("Retry-After", "0"))
            except (TypeError, ValueError):
                ra = 0
            pausa = max(attesa, ra)
            print("   HTTP %d (server sovraccarico/limite), tentativo %d/%d, nuovo tentativo tra %ds"
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--operazione", choices=["vendita", "affitto", "entrambe"], default="entrambe")
    p.add_argument("--modello", default="gpt-6-luna")
    p.add_argument("--pausa", type=float, default=6.0, help="secondi tra una chiamata e l'altra")
    p.add_argument("--batch", type=int, default=1, help="annunci per chiamata (1 = uno alla volta)")
    p.add_argument("--prova", type=int, help="elabora solo N annunci (da classificare)")
    p.add_argument("--stima", action="store_true", help="conta soltanto, senza chiamare ChatGPT")
    p.add_argument("--rifai", action="store_true", help="ignora l'archivio e riclassifica tutto")
    a = p.parse_args()

    ops = ["vendita", "affitto"] if a.operazione == "entrambe" else [a.operazione]
    chiave = os.environ.get("OPENAI_API_KEY", "").strip()
    if not chiave and not a.stima:
        print("Manca la chiave. In Anaconda Prompt:  set OPENAI_API_KEY=la_tua_chiave"); return 1

    cache = carica_cache()
    lavoro, tabelle = [], {}
    for op in ops:
        f = os.path.join(AQUI, "annunci_%s_spese.csv" % op)
        if not os.path.exists(f):
            print("File assente: annunci_%s_spese.csv" % op); continue
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
    tot = sum(len(t[1]) for t in tabelle.values())
    print("Annunci totali: %d | da classificare: %d | gia' in archivio: %d" % (tot, len(lavoro), tot - len(lavoro)))
    if a.stima:
        print("Chiamate previste a OpenAI:", len(lavoro)); return 0
    if a.prova:
        lavoro = lavoro[:a.prova]

    def reg(op, r, h, cat, motivo):
        cache[(op, r["id_annuncio"])] = dict(id_annuncio=r["id_annuncio"], operazione=op, hash=h, categoria=cat,
                                             rif="SI" if cat == "immobile" else "NO", motivo=motivo, modello=a.modello)
    L.elabora(lavoro, max(1, a.batch), a.pausa,
              lambda op, r: chiama(a.modello, chiave, op, testo_annuncio(r)),
              lambda op, voci: chiama(a.modello, chiave, op, L.testo_lotto([(i, testo_annuncio(r)) for i, r in voci]), lotto=True),
              reg, lambda: salva_cache(cache), CATEGORIE)
    salva_cache(cache)

    for op, (campi, righe) in tabelle.items():
        n = {"SI": 0, "NO": 0, "": 0}
        for r in righe:
            c = cache.get((op, r["id_annuncio"]))
            r["rif_openai"] = c["rif"] if c else ""
            r["categoria_openai"] = c["categoria"] if c else ""
            r["motivo_openai"] = c["motivo"] if c else ""
            n[r["rif_openai"]] += 1
        dst = os.path.join(AQUI, "annunci_%s_spese.csv" % op)
        bk = os.path.join(AQUI, "_to_delete")
        os.makedirs(bk, exist_ok=True)
        shutil.copyfile(dst, os.path.join(bk, "annunci_%s_spese_prima_openai.csv" % op))
        tmp = dst + ".tmp"
        with open(tmp, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campi + NUOVE, delimiter=";")
            w.writeheader(); w.writerows(righe)
        os.replace(tmp, dst)
        confr = sum(1 for r in righe if r.get("rif") and r["rif_openai"] and r["rif"] != r["rif_openai"])
        print("%s: rif_openai SI %d | NO %d | non ancora classificati %d | discordanti da rif: %d  ->  %s"
              % (op, n["SI"], n["NO"], n[""], confr, os.path.basename(dst)))
    return 0

if __name__ == "__main__":
    sys.exit(main())
