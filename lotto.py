# -*- coding: utf-8 -*-
"""Supporto comune agli script classifica_*.py: invio di PIU' annunci (default 30) in una sola chiamata."""
import json, time

SUFFISSO = """

MODALITA' LOTTO: ricevi piu' annunci, ciascuno introdotto da una riga '### ANNUNCIO <id>'.
Classifica OGNI annuncio in modo INDIPENDENTE, usando solo i dati di quell'annuncio (non mescolare dati di annunci diversi).
Rispondi con un JSON {"risultati": [ {"id": "<id>", "categoria": "...", "motivo": "..."} ]} con UN elemento per
ciascun annuncio ricevuto, riportando esattamente lo stesso id."""


def schema_lotto(categorie, maiuscolo):
    S, O, A = ("STRING", "OBJECT", "ARRAY") if maiuscolo else ("string", "object", "array")
    voce = {"type": O, "properties": {"id": {"type": S}, "categoria": {"type": S, "enum": categorie},
                                      "motivo": {"type": S}}, "required": ["id", "categoria", "motivo"]}
    radice = {"type": O, "properties": {"risultati": {"type": A, "items": voce}}, "required": ["risultati"]}
    if not maiuscolo:
        voce["additionalProperties"] = False
        radice["additionalProperties"] = False
    return radice


def testo_lotto(voci):
    """voci = [(id, testo_annuncio)]"""
    return "\n\n".join("### ANNUNCIO %s\n%s" % (i, t) for i, t in voci)


def leggi_lotto(testo_json, ids, categorie):
    """Ritorna {id: (categoria, motivo)} solo per gli id attesi e validi."""
    try:
        elenco = json.loads(testo_json).get("risultati", [])
    except (ValueError, AttributeError):
        return {}
    out = {}
    for x in elenco:
        if not isinstance(x, dict):
            continue
        i = str(x.get("id", "")).strip()
        if i in ids and i not in out:
            cat = x.get("categoria", "incerto")
            out[i] = (cat if cat in categorie else "incerto", " ".join(str(x.get("motivo", "")).split()))
    return out


def elabora(lavoro, n, pausa, singola, lotto, registra, salva, categorie):
    """lavoro = [(op, riga, hash)]. singola(op, riga)->(cat,motivo)|None ; lotto(op, [(id, riga)])->testo_json|None.
    registra(op, riga, hash, cat, motivo)."""
    tot, fatti, i = len(lavoro), 0, 0
    while i < tot:
        op = lavoro[i][0]
        j = i
        while j < tot and j - i < n and lavoro[j][0] == op:
            j += 1
        blocco = lavoro[i:j]
        i = j
        ids = {r["id_annuncio"] for _, r, _ in blocco}
        raw = lotto(op, [(r["id_annuncio"], r) for _, r, _ in blocco]) if len(blocco) > 1 else None
        esiti = leggi_lotto(raw, ids, categorie) if raw else {}
        if len(blocco) > 1 and raw is None:
            print("[%d/%d] lotto di %d annunci senza risposta, salto (riprova al prossimo lancio)" % (fatti, tot, len(blocco)))
            fatti += len(blocco)
            continue
        for op_, r, h in blocco:
            fatti += 1
            e = esiti.get(r["id_annuncio"])
            via = "lotto"
            if e is None:
                e = singola(op_, r); via = "singola"
            if e is None:
                print("[%d/%d] %s: nessuna risposta, salto (riprova al prossimo lancio)" % (fatti, tot, r["id_annuncio"]))
                continue
            registra(op_, r, h, e[0], e[1])
            print("[%d/%d] %s %s -> %s | %s | %s (%s)" % (fatti, tot, op_, r["id_annuncio"], "SI" if e[0] == "immobile" else "NO", e[0], e[1], via))
        salva()
        time.sleep(pausa)
