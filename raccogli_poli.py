#!/usr/bin/env python3
"""
raccogli_poli.py - Passo 1 automatico: ospedali + università di Medicina per città.

Per ogni città in citta.txt (una per riga):
  1. ricava da Nominatim i CONFINI del comune (solo il comune, non la provincia)
  2. scarica da Overpass API ospedali e sedi universitarie sanitarie
  3. scarta uffici ASL/ULSS, Croce Rossa, oggetti senza nome; aggiunge politecnici, IRCCS e grandi università private
  4. raggruppa i punti vicini (<= 400 m) in un unico polo
  5. assegna un peso (alto/medio/basso)
Poi unisce i poli verificati a mano in poli_manuali.csv (hanno la precedenza)
e scrive poli.csv, già pronto per genera_aree.py.

Solo libreria standard Python 3, nessuna installazione.

Uso:
    python raccogli_poli.py                        # legge citta.txt
    python raccogli_poli.py --citta Padova Bologna Napoli
    python raccogli_poli.py --italia               # tutta Italia (richiesta pesante)
    python raccogli_poli.py --min-peso basso       # tiene anche i poli di peso basso (default: medio)
    python raccogli_poli.py --da-json risposta.json --citta Padova   # prova offline
"""
import argparse, csv, json, math, os, re, sys, time
import urllib.parse, urllib.request

UA = "gdc-immobiliare/1.0 (uso personale)"
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]
NOMINATIM = "https://nominatim.openstreetmap.org/search"
RAGGIO_CLUSTER_M = 400
RE_MED = "medicin|policlinic|salute|chirurg|odontoiat|biomedic|sanit|infermier|politecnic"
# IRCCS: sigla + grandi nomi noti che spesso in OSM non riportano la sigla
RE_IRCCS_NOME = ("irccs|humanitas|san raffaele|istituto europeo di oncologia|don gnocchi|"
                 "multimedica|galeazzi|policlinico san donato|auxologico|casa sollievo della sofferenza|"
                 "policlinico gemelli|fondazione policlinico|negrar|sacro cuore don calabria|"
                 "ospedale pediatrico bambino|mondino|neuromed|centro cardiologico monzino|"
                 "istituto clinico|istituto neurologico|istituto oncologico|istituto ortopedico|"
                 "mater olbia|san camillo|"
                 # pubblici (spesso senza la sigla IRCCS in OSM)
                 "gaslini|burlo garofolo|regina elena|spallanzani|bambino ges|"
                 "policlinico san martino|ca. granda|ospedale maggiore policlinico|"
                 "istituto nazionale|istituto tumori|pascale|ismett|centro di riferimento oncologico|"
                 "inrca|istituto scientifico|sant.orsola|ospedale pediatrico meyer|ospedale meyer|"
                 "fondazione santa lucia|stella maris|policlinico di milano")

RE_ADMIN = re.compile(r"\b(USL|ULSS|ASL|AUSL|ASST|ASUR|ASP|AST|distretto|croce rossa|"
                      r"misericordia|pubblica assistenza|guardia medica|veterinar)", re.I)
RE_HOSP = re.compile(r"ospedal|policlinic|presidio|clinica|casa di cura|istituto|irccs", re.I)
RE_IRCCS = re.compile(RE_IRCCS_NOME, re.I)
RE_PRIVATO = re.compile(r"humanitas|san raffaele|istituto europeo di oncologia|\bieo\b|don gnocchi|"
                        r"multimedica|galeazzi|san donato|auxologico|casa sollievo|gemelli|"
                        r"negrar|don calabria|monzino|mater olbia|san camillo|fatebenefratelli", re.I)
RE_PUBBLICO = re.compile(r"gaslini|burlo|regina elena|spallanzani|rizzoli|policlinico san martino|"
                         r"ca. granda|ospedale maggiore policlinico|istituto nazionale|istituto tumori|"
                         r"pascale|ismett|centro di riferimento oncologico|inrca|sant.orsola|meyer|"
                         r"istituto neurologico|istituto oncologico|istituto scientifico|"
                         r"policlinico di milano|stella maris|sacco", re.I)
RE_UNIV_PRIV_NOME = ("bocconi|cattolica|luiss|lumsa|iulm|humanitas univ|vita-salute|vita salute|"
                     "bio-?medico|unicamillus|saint camillus|europea di roma|cusano|liuc|"
                     "carlo cattaneo|suor orsola|kore di enna|universit. kore|jean monnet|"
                     "internazionali di roma|link campus|e-?campus|iusve")
RE_UNIV_PRIV = re.compile(RE_UNIV_PRIV_NOME, re.I)
RE_UNIV_PRIV_MED = re.compile(r"humanitas univ|vita-salute|vita salute|bio-?medico|unicamillus|"
                              r"saint camillus|kore di enna|universit. kore|jean monnet", re.I)
RE_POLITECNICO = re.compile(r"politecnic", re.I)
RE_UNIV_OSP = re.compile(r"policlinic|universitari|azienda ospedal\w*[- ]universit|aou\b|"
                         r"ospedale[- ]universit", re.I)
RE_ALTO = re.compile(r"irccs|ospedale maggiore|ospedali riuniti|grande ospedale", re.I)
RE_MEDIO = re.compile(r"ospedal|presidio", re.I)
RE_BASSO = re.compile(r"militare|casa di cura|centro medico|poliambulatorio", re.I)
RE_NOISE = re.compile(r"ambulatori|centro (prelievi|diurno|diagnostico|medico|sanitario)|"
                      r"casa di riposo|\bRSA\b|residenza|medicina di gruppo|punto medico|"
                      r"poliambulatori|casa di comunit|socio[- ]?sanit|hospice|salute mentale|"
                      r"s\.?r\.?l\b|army|dialisi|riabilitaz", re.I)
RANK = {"alto": 3, "medio": 2, "basso": 1}

def hav(a, b):
    R = 6371000.0
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2*R*math.asin(math.sqrt(h))

# ---------- rete ----------
def http(url, data=None, tentativi=4, pausa=8):
    ultimo = None
    for i in range(tentativi):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read().decode("utf-8")
        except Exception as e:
            ultimo = e
            time.sleep(pausa * (i + 1))
    raise RuntimeError("richiesta fallita: %s (%s)" % (url[:80], ultimo))

def area_citta(citta):
    """Restituisce l'id area Overpass del COMUNE (non della provincia)."""
    q = urllib.parse.urlencode({"q": citta, "format": "json", "limit": 8,
                                "countrycodes": "it", "featuretype": "city"})
    res = json.loads(http(NOMINATIM + "?" + q))
    time.sleep(1.2)  # limite Nominatim: 1 richiesta/secondo
    for r in res:
        if r.get("osm_type") == "relation" and r.get("class") == "boundary" \
                and int(r.get("place_rank", 0)) >= 14:
            return 3600000000 + int(r["osm_id"]), r.get("display_name", citta)
    raise RuntimeError("comune non trovato su OpenStreetMap: %s "
                       "(prova con il nome ufficiale, es. 'Padova')" % citta)

def query_overpass(area_filter):
    ar, flt = area_filter
    ql = ('[out:json][timeout:180];%s(\n'
          '  nwr["amenity"="hospital"]%s;\n'
          '  nwr["amenity"~"university|college"]["name"~"%s|%s",i]%s;\n'
          '  nwr["amenity"="clinic"]["name"~"%s",i]%s;\n'
          '  nwr["healthcare"]["amenity"!~"hospital|clinic|university|college"]["name"~"%s",i]%s;\n'
          ');\nout center tags;') % (ar, flt, RE_MED, RE_UNIV_PRIV_NOME, flt,
                                     RE_IRCCS_NOME, flt, RE_IRCCS_NOME, flt)
    data = urllib.parse.urlencode({"data": ql}).encode()
    for url in OVERPASS:
        try:
            return json.loads(http(url, data=data))
        except Exception as e:
            print("  Overpass non risponde su %s: %s" % (url, e), file=sys.stderr)
    raise RuntimeError("nessun server Overpass raggiungibile")

# ---------- pulizia ----------
def classifica(tags):
    nome = (tags.get("name") or tags.get("official_name") or "").strip()
    if not nome:
        return None
    amenity = tags.get("amenity", "")
    privato = tags.get("operator:type") in ("private", "private_nonprofit", "religious") \
        or bool(RE_PRIVATO.search(nome))
    if amenity in ("university", "college"):
        if RE_POLITECNICO.search(nome):
            return nome, "politecnico", "medio"
        if RE_UNIV_PRIV.search(nome):
            return nome, "università privata", "alto" if RE_UNIV_PRIV_MED.search(nome) else "medio"
        return nome, "università (area sanitaria)", "medio"
    # IRCCS (anche se in OSM non è etichettato come ospedale)
    if RE_IRCCS.search(nome) and not RE_ADMIN.search(nome):
        if amenity not in ("hospital", "clinic") and not tags.get("healthcare"):
            return None
        if privato:
            tipo = "IRCCS privato"
        elif tags.get("operator:type") == "public" or RE_PUBBLICO.search(nome):
            tipo = "IRCCS pubblico"
        else:
            tipo = "IRCCS"
        return nome, tipo, "alto"
    if amenity != "hospital":
        return None
    if RE_ADMIN.search(nome) and not RE_HOSP.search(nome):
        return None
    if tags.get("emergency") == "ambulance_station" and not tags.get("beds"):
        return None
    if RE_NOISE.search(nome) and not RE_UNIV_OSP.search(nome) \
            and tags.get("emergency") not in ("yes", "designated"):
        return None
    try:
        beds = int(re.sub(r"\D", "", tags.get("beds", "")) or 0)
    except ValueError:
        beds = 0
    if tags.get("emergency") in ("yes", "designated") or beds >= 300 \
            or RE_UNIV_OSP.search(nome) or RE_ALTO.search(nome):
        peso = "alto"
    elif RE_MEDIO.search(nome) or beds >= 100:
        peso = "medio"
    else:
        peso = "basso"
    if RE_BASSO.search(nome) and tags.get("emergency") not in ("yes", "designated"):
        peso = "basso"
    tipo = "università+ospedale" if RE_UNIV_OSP.search(nome) else "ospedale"
    return nome, tipo, peso

def estrai(risposta, citta_default):
    punti = []
    for el in risposta.get("elements", []):
        tags = el.get("tags", {})
        c = classifica(tags)
        if not c:
            continue
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        nome, tipo, peso = c
        punti.append({"nome": nome, "tipo": tipo, "peso": peso,
                      "citta": citta_default or tags.get("addr:city", ""),
                      "lat": float(lat), "lon": float(lon), "osm": el.get("id")})
    return punti

def raggruppa(punti):
    """Unisce i punti entro 400 m; vince il peso più alto (poi università+ospedale)."""
    ordine = sorted(punti, key=lambda p: (-RANK[p["peso"]],
                                          p["tipo"] != "università+ospedale", p["nome"]))
    poli = []
    for p in ordine:
        for g in poli:
            if hav((g["lat"], g["lon"]), (p["lat"], p["lon"])) <= RAGGIO_CLUSTER_M:
                g["n"] += 1
                g["altri"].append(p["nome"])
                break
        else:
            poli.append(dict(p, n=1, altri=[]))
    return poli

def unisci_manuali(poli, path):
    if not os.path.exists(path):
        return poli, []
    with open(path, encoding="utf-8", newline="") as f:
        manuali = list(csv.DictReader(f, delimiter=";"))
    for m in manuali:
        pos = (float(m["lat"]), float(m["lon"]))
        poli = [p for p in poli if hav((p["lat"], p["lon"]), pos) > RAGGIO_CLUSTER_M]
    return poli, manuali

def scrivi_poli(tutti, a, P, silenzioso=False):
    poli = raggruppa(tutti)
    poli, manuali = unisci_manuali(poli, P(a.manuali))
    poli = [p for p in poli if RANK[p["peso"]] >= RANK[a.min_peso]]
    poli.sort(key=lambda p: (p["citta"], -RANK[p["peso"]], p["nome"]))
    contatori = {}
    righe = []
    for m in manuali:
        righe.append([m["id"], m["nome"], m["tipo"], m["citta"], m["lat"], m["lon"],
                      m["peso"], m["fonte"], m["note"]])
    for p in poli:
        k = (p["citta"] or "IT")[:3].upper()
        contatori[k] = contatori.get(k, 0) + 1
        note = "OSM %s" % p["osm"]
        if p["n"] > 1:
            note += "; unisce %d punti: %s" % (p["n"], ", ".join(p["altri"][:4]))
        righe.append(["%s%02d" % (k, contatori[k]), p["nome"], p["tipo"], p["citta"],
                      "%.6f" % p["lat"], "%.6f" % p["lon"], p["peso"], "OSM", note])
    with open(P(a.out), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "nome", "tipo", "citta", "lat", "lon", "peso", "fonte", "note"])
        w.writerows(righe)
    if not silenzioso:
        print("Scritti %d poli in %s (di cui %d manuali)" % (len(righe), a.out, len(manuali)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--citta", nargs="*")
    ap.add_argument("--file", default="citta.txt")
    ap.add_argument("--italia", action="store_true")
    ap.add_argument("--min-peso", choices=["basso", "medio", "alto"], default="medio")
    ap.add_argument("--manuali", default="poli_manuali.csv")
    ap.add_argument("--out", default="poli.csv")
    ap.add_argument("--da-json", help="risposta Overpass salvata (prova offline)")
    a = ap.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    P = lambda x: x if os.path.isabs(x) else os.path.join(base, x)

    tutti = []
    if a.italia:
        print("Scarico tutta Italia (può richiedere diversi minuti)...")
        r = query_overpass(('area["ISO3166-1"="IT"][admin_level=2]->.a;', "(area.a)"))
        tutti += estrai(r, "")
    else:
        citta = a.citta
        if not citta:
            with open(P(a.file), encoding="utf-8") as f:
                citta = [x.strip() for x in f if x.strip() and not x.startswith("#")]
        errori = []
        t0 = time.time()
        for i, c in enumerate(citta, 1):
            print("[%d/%d] %s  (%.0f s)" % (i, len(citta), c, time.time() - t0))
            try:
                if a.da_json:
                    with open(P(a.da_json), encoding="utf-8") as f:
                        r = json.load(f)
                else:
                    aid, disp = area_citta(c)
                    print("  area comune:", disp[:90])
                    r = query_overpass(("area(%d)->.a;" % aid, "(area.a)"))
                    time.sleep(3)  # cortesia verso Overpass
                tutti += estrai(r, c)
                scrivi_poli(tutti, a, P, silenzioso=True)  # poli.csv sempre aggiornato
            except Exception as e:
                print("  ERRORE su %s: %s" % (c, e), file=sys.stderr)
                errori.append(c)
        if errori:
            with open(P("citta_errori.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(errori) + "\n")
            print("\nCittà non riuscite (%d), salvate in citta_errori.txt: %s"
                  % (len(errori), ", ".join(errori)))
    scrivi_poli(tutti, a, P)

if __name__ == "__main__":
    main()
