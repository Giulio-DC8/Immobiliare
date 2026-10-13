#!/usr/bin/env python3
"""
estrai_poli_pbf.py - Passo 1 offline: poli sanitari/universitari da un estratto OSM (PBF).

Legge un file .osm.pbf (Geofabrik) e produce poli.csv senza usare server pubblici.
Riusa le stesse regole di raccogli_poli.py (filtri, pesi, IRCCS, politecnici, università
private, raggruppamento entro 400 m, unione con poli_manuali.csv).

Requisiti:   pip install osmium
Uso:
    python estrai_poli_pbf.py --pbf dati\\italy-latest.osm.pbf
    python estrai_poli_pbf.py --pbf dati\\nord-est-latest.osm.pbf        # prova su una macroarea
    python estrai_poli_pbf.py --pbf dati\\italy-latest.osm.pbf --solo-citta citta.txt
    python estrai_poli_pbf.py --test                                       # verifica la logica

La città è assegnata da addr:city, altrimenti dal centro abitato più vicino (place=city/town,
oppure village se il centro più vicino dista oltre 8 km): è un'approssimazione, usata solo
per etichetta e codice del polo. Le coordinate sono quelle reali.
"""
import argparse, collections, math, os, re, sys, time
from types import SimpleNamespace

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import raccogli_poli as rp

RE_UNIV_NOME = re.compile(rp.RE_MED + "|" + rp.RE_UNIV_PRIV_NOME, re.I)
CELLA = 0.2  # gradi, per l'indice spaziale dei centri abitati


class Indice:
    """Centri abitati con ricerca del più vicino."""
    def __init__(self):
        self.celle = {"city": {}, "village": {}}

    def aggiungi(self, nome, lat, lon, place):
        grp = "city" if place in ("city", "town") else "village"
        self.celle[grp].setdefault((int(lat // CELLA), int(lon // CELLA)), []).append((nome, lat, lon))

    def piu_vicino(self, lat, lon, grp, raggio_celle=3):
        ci, cj = int(lat // CELLA), int(lon // CELLA)
        migliore = (None, 1e18)
        for di in range(-raggio_celle, raggio_celle + 1):
            for dj in range(-raggio_celle, raggio_celle + 1):
                for nome, la, lo in self.celle[grp].get((ci + di, cj + dj), ()):
                    d = rp.hav((lat, lon), (la, lo))
                    if d < migliore[1]:
                        migliore = (nome, d)
        return migliore

    def citta(self, lat, lon):
        nome, d = self.piu_vicino(lat, lon, "city")
        if nome and d <= 8000:
            return nome
        n2, d2 = self.piu_vicino(lat, lon, "village")
        if n2 and (nome is None or d2 < d):
            return n2
        return nome or ""


class Raccolta:
    def __init__(self):
        self.punti = []
        self.indice = Indice()
        self.contatori = {"nodi_utili": 0, "aree_utili": 0}
        self.t0 = time.time()
        self.ultimo = None          # (nome, lat, lon) dell'ultimo polo trovato
        self.dettaglio = False      # True: stampa ogni polo trovato

    def battito(self, cosa, n):
        """Messaggio di avanzamento durante la lettura del PBF."""
        s = int(time.time() - self.t0)
        msg = "[%02d:%02d] letti %s %s | poli trovati finora: %d" % (
            s // 60, s % 60, "{:,}".format(n).replace(",", "."), cosa, len(self.punti))
        if self.ultimo:
            nome, la, lo = self.ultimo
            vicino = self.indice.citta(la, lo) or "?"
            msg += " | ultimo: %s (%.4f, %.4f) vicino a %s" % (nome[:45], la, lo, vicino)
        print(msg, flush=True)

    @staticmethod
    def candidato(tags):
        nome = tags.get("name") or tags.get("official_name")
        if not nome:
            return False
        am = tags.get("amenity", "")
        if am == "hospital":
            return True
        if am in ("university", "college"):
            return bool(RE_UNIV_NOME.search(nome))
        if am == "clinic" or "healthcare" in tags:
            return bool(rp.RE_IRCCS.search(nome))
        return False

    def elemento(self, tags, lat, lon, osm_id):
        """tags: dict. Registra centri abitati e poli candidati."""
        place = tags.get("place")
        if place in ("city", "town", "village") and tags.get("name"):
            self.indice.aggiungi(tags["name"], lat, lon, place)
            return
        if not self.candidato(tags):
            return
        c = rp.classifica(tags)
        if not c:
            return
        nome, tipo, peso = c
        self.punti.append({"nome": nome, "tipo": tipo, "peso": peso,
                           "citta": tags.get("addr:city", ""),
                           "lat": float(lat), "lon": float(lon), "osm": osm_id})
        self.ultimo = (nome, float(lat), float(lon))
        if self.dettaglio:
            print("  + [%s/%s] %s (%.4f, %.4f)" % (tipo, peso, nome[:60], float(lat), float(lon)), flush=True)

    def etichetta_citta(self):
        for p in self.punti:
            if not p["citta"]:
                p["citta"] = self.indice.citta(p["lat"], p["lon"])


def leggi_pbf(percorso, racc):
    try:
        import osmium
    except ImportError:
        sys.exit("Manca la libreria osmium. Installa con:  pip install osmium")

    def tags_dict(tl):
        return {t.k: t.v for t in tl}

    class H(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.n_nodi = 0
            self.n_aree = 0

        def node(self, n):
            self.n_nodi += 1
            if self.n_nodi % 10000000 == 0:
                racc.battito("nodi", self.n_nodi)
            tl = n.tags
            if not (tl.get("amenity") or tl.get("healthcare") or tl.get("place")):
                return
            if not n.location.valid():
                return
            racc.elemento(tags_dict(tl), n.location.lat, n.location.lon, "n%d" % n.id)

        def area(self, a):
            self.n_aree += 1
            if self.n_aree % 200000 == 0:
                racc.battito("edifici/aree", self.n_aree)
            tl = a.tags
            if not (tl.get("amenity") or tl.get("healthcare")):
                return
            lat = lon = None
            try:
                for ring in a.outer_rings():
                    pts = [(nr.lat, nr.lon) for nr in ring]
                    if pts:
                        lat = sum(p[0] for p in pts) / len(pts)
                        lon = sum(p[1] for p in pts) / len(pts)
                        break
            except Exception:
                return
            if lat is None:
                return
            pref = "w" if a.from_way() else "r"
            racc.elemento(tags_dict(tl), lat, lon, "%s%d" % (pref, a.orig_id()))

    print("Lettura di %s (può richiedere diversi minuti)..." % percorso)
    print("Avanzamento ogni 10 milioni di nodi / 200.000 edifici. Ordine di lettura: per ID, non per città.", flush=True)
    t0 = time.time()
    racc.t0 = t0
    H().apply_file(percorso, locations=True)
    print("Lettura completata in %.0f s" % (time.time() - t0))


def test():
    """Verifica la logica con dati simulati (senza file PBF)."""
    racc = Raccolta()
    for nome, la, lo in (("Padova", 45.4064, 11.8768), ("Rovigo", 45.0698, 11.7903)):
        racc.elemento({"place": "city", "name": nome}, la, lo, "n0")
    dati = [
        ({"amenity": "hospital", "name": "Azienda Ospedaliera di Padova", "emergency": "yes"}, 45.4035, 11.8907),
        ({"amenity": "hospital", "name": "Ospedale Giustinianeo"}, 45.4036, 11.8868),
        ({"amenity": "hospital", "name": "Ospedale Sant'Antonio", "emergency": "yes"}, 45.3927, 11.8918),
        ({"amenity": "hospital", "name": "ULSS 6 Euganea"}, 45.4129, 11.8844),
        ({"amenity": "hospital", "name": "Ambulatorio di Salute Mentale"}, 45.33, 11.78),
        ({"amenity": "university", "name": "Università Commerciale Luigi Bocconi"}, 45.4500, 9.1900),
        ({"amenity": "university", "name": "Dipartimento di Ingegneria Civile"}, 45.4072, 11.8936),
        ({"amenity": "hospital", "name": "IRCCS Ospedale San Raffaele"}, 45.5058, 9.2650),
        ({"amenity": "hospital", "name": "Ospedale di Rovigo", "emergency": "yes"}, 45.0700, 11.7900),
        ({"healthcare": "hospital", "name": "Istituto Giannina Gaslini", "amenity": "clinic"}, 44.4260, 8.9780),
    ]
    for i, (t, la, lo) in enumerate(dati):
        racc.elemento(t, la, lo, "w%d" % i)
    racc.etichetta_citta()
    out = "/tmp/test_poli_pbf.csv" if os.name != "nt" else os.path.join(BASE, "test_poli_pbf.csv")
    args = SimpleNamespace(manuali=os.path.join(BASE, "poli_manuali.csv"), min_peso="medio", out=out)
    rp.scrivi_poli(racc.punti, args, lambda x: x)
    print(open(out, encoding="utf-8").read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", default=os.path.join("dati", "italy-latest.osm.pbf"))
    ap.add_argument("--solo-citta", help="file con le città da tenere (es. citta.txt)")
    ap.add_argument("--min-peso", choices=["basso", "medio", "alto"], default="medio")
    ap.add_argument("--manuali", default="poli_manuali.csv")
    ap.add_argument("--out", default="poli.csv")
    ap.add_argument("--dettaglio", action="store_true", help="stampa ogni polo trovato durante la lettura")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        return test()
    P = lambda x: x if os.path.isabs(x) else os.path.join(BASE, x)
    racc = Raccolta()
    racc.dettaglio = a.dettaglio
    leggi_pbf(P(a.pbf), racc)
    print("Assegno la città a %d poli trovati..." % len(racc.punti), flush=True)
    racc.etichetta_citta()
    punti = racc.punti
    conta = collections.Counter(p["citta"] or "(senza città)" for p in punti)
    print("Poli per città, prime 20: " + ", ".join("%s %d" % (c, n) for c, n in conta.most_common(20)), flush=True)
    if a.solo_citta:
        with open(P(a.solo_citta), encoding="utf-8") as f:
            ok = {x.split(",")[0].strip().casefold() for x in f
                  if x.strip() and not x.startswith("#")}
        punti = [p for p in punti if p["citta"].casefold() in ok]
        print("Filtrati su %d città: restano %d punti" % (len(ok), len(punti)))
        trovate = {p["citta"].casefold() for p in punti}
        senza = sorted(c for c in ok if c not in trovate)
        if senza:
            print("Città di %s senza alcun polo trovato (%d): %s" % (a.solo_citta, len(senza), ", ".join(senza)), flush=True)
    rp.scrivi_poli(punti, a, P)


if __name__ == "__main__":
    main()
