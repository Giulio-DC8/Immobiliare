# GDC Immobiliare – Immobili vicino a poli sanitari e universitari

**Obiettivo.** Individuare appartamenti e villette in **vendita** (poi **affitto**) nei dintorni di
ospedali, IRCCS, policlinici universitari, facoltà di Medicina, politecnici e grandi università private,
per acquisirli e riutilizzarli nel canale affitto o vendita.

**Regole di lavoro.** Si agisce solo su comando e conferma dell'utente. Gli script girano sul PC
dell'utente (Anaconda Prompt), nella cartella `C:\Users\giuli\Desktop\GDC Immobiliare`.
La chiave Google non va mai scritta in file o chat: solo come variabile d'ambiente.

> **Idealista:** ha protezioni anti-bot e i termini d'uso vietano lo scraping. Non si aggirano i blocchi.
> La raccolta annunci avviene in modalità assistita (pagine salvate dal proprio browser) oppure con API
> ufficiale / fornitori autorizzati.

---

## Procedura in sintesi

### Passo 1 – Elenco ufficiale dei poli  ✅ fatto
1. Province da analizzare scelte per iscritti fuori regione: `province_iscritti.csv` + `filtra_citta.py --soglia N` -> `citta.txt`.
2. Elenchi (Perplexity) in `dati\`: A IRCCS, B AOU/policlinici, C Facoltà di Medicina, D Politecnici, E Grandi università private.
3. Unificazione in **`poli_ufficiali.csv`** (220 righe; id U001…; colonne `categoria`, `ente_o_ateneo`, `indirizzo`, ecc.).
4. Backup dei file sostituiti in `_to_delete\` (la cartella si può svuotare a mano).

### Passo 2 – Coordinate precise con Google Maps  ✅ fatto
1. Google Cloud: progetto "Maps - GDC Immobiliare", fatturazione (prova gratuita), abilitare **Places API (New)**.
   Chiave API limitata alle API necessarie; budget con avvisi (es. 10 €).
2. Input: `tuoi_indirizzi.csv` con tre colonne **`nome`, `ENTE`, `provincia`** (220 righe).
3. Anaconda Prompt, nella cartella del progetto:
   ```
   set GOOGLE_MAPS_API_KEY=la_chiave
   python geocoda_google.py --stima
   python geocoda_google.py --prova 5
   python geocoda_google.py            (oppure --riprendi)
   ```
   `geocoda_google.py` cerca su Google Places in fino a 3 tentativi ("nome, ente, provincia" -> "ente, provincia" ->
   "ente ospedale/università, provincia"), sceglie il candidato con provincia corretta e tipo pertinente e lo confronta
   con le coordinate di riferimento di `poli_ufficiali.csv` (se >1 km: "da verificare").
   Output: `tuoi_indirizzi_google_nome.csv` (colonne `query_usata`, `altri_candidati`, `controllo`).
4. Verifica a mano delle righe "da verificare" (52 su 220) e integrazione in `poli_ufficiali.csv`
   (colonne `lat_google`, `lon_google`, `nome_google`, `indirizzo_google`, `place_id_google`, `controllo_google`;
   coordinate precedenti in `lat_google_prec`, `lon_google_prec`, `spostamento_m`).
   Stato: 216 validati, 4 esclusi (università telematiche Mercatorum e Pegaso).
   Limite noto: per alcuni enti (nomi molto lunghi) Google sceglie un luogo sbagliato; i candidati scartati sono in `altri_candidati`.

### Passo 3 – Aree di ricerca Idealista  ✅ fatto
```
python genera_aree_ufficiali.py                          -> aree_vendita.csv  (152 aree)
python genera_aree_ufficiali.py --operazione affitto     -> aree_affitto.csv  (152 aree)
```
- Usa i poli "validato" con coordinate Google; raggruppa i poli entro 400 m (`--cluster`).
- Cerchio di raggio 1000 m (`--raggio`), poligono a 14 vertici, URL con `shape=((…))`.
- Filtri vendita: `con-prezzo_120000,dimensione-max_60,nuova-costruzione,buono-stato,aste_no,senza-inquilini` (massimo 120.000 € e 60 m²) (in `genera_aree.py`, `FILTRI_VENDITA`).
- Filtri affitto: `con-dimensione-max_60,arredamento_ammobiliato,nuova-costruzione,buono-stato` (`FILTRI_AFFITTO`; massimo 60 m², ammobiliato, nessun limite di prezzo).
- Controllo: aprire a mano 2–3 URL (es. Padova, Roma) e verificare che il poligono sia nel punto giusto.

### Passo 4 – Raccolta annunci  ✅ metodo attivo (raccolta assistita)
Idealista blocca l'automazione e vieta lo scraping: gli annunci si raccolgono dalle pagine che l'utente apre nel proprio browser.
1. Aprire l'URL di un'area (`aree_vendita.csv` / `aree_affitto.csv`), cliccare "Vai alla lista" (vista **elenco**, non mappa) e scorrere fino in fondo.
2. **Metodo consigliato (console):** F12 -> Console (la prima volta digitare `allow pasting`), incollare il contenuto di
   `estrai_annunci_console.js` e premere Invio: copia negli appunti gli annunci visibili. Incollare in un file `.txt` in
   `dati\pagine\` (es. `FERRARA_affitto.txt`, un nome diverso per operazione e per pagina successiva).
3. Metodo alternativo: salvare la pagina (Ctrl+S, solo HTML) in `dati\pagine\`. Se Avast o altri programmi di sicurezza
   rimuovono il file (capita con Idealista), usare il metodo della console.
4. Eseguire `python leggi_annunci.py`: legge i `.html` e i `.txt` di `dati\pagine\`, abbina l'area, calcola il prezzo al mq e aggiorna
   `annunci_vendita.csv` e `annunci_affitto.csv` senza duplicati per ID (conserva prima rilevazione, primo prezzo e gli annunci non
   piu' visti, colonna `presente_ultimo_lancio`).
5. Se un'area ha piu' annunci di quelli visibili, ripetere sulle pagine successive.

Alternative: API ufficiale di Idealista (accreditamento) o dati da fornitori autorizzati; aste giudiziarie (PVP).

### Passo 5 – Pulizia e analisi  ⏳ da fare
- Deduplicare per ID; distanza in linea d'aria (haversine) dal polo più vicino e tipo di polo.
- €/mq; ordinamento per distanza e €/mq.
- Con i dati di affitto sulla stessa area: **rendimento lordo = canone annuo / prezzo di acquisto**.
- Ripetere periodicamente per nuovi annunci, ribassi, annunci spariti.

### Passo 6 – Completamenti facoltativi  ⏳ da valutare
- Ospedali generali non IRCCS/AOU (es. Sant'Antonio di Padova): ricerca Google Places per provincia.
- Estrazione OpenStreetMap (`estrai_poli_pbf.py` su `dati\italy-261002.osm.pbf`): utile come confronto, meno accurata di Google.
- Pesi differenziati per categoria (colonna `categorie` in `aree_*.csv`; oggi tutte le aree sono `alto`).
- Posti letto / iscritti per ponderare i poli (oggi disponibili per sole 20 righe).

---

## Criteri di valutazione
- Affitto a studenti di Medicina: camere e bilocali a pochi minuti da aule e policlinico, contratti 6–12 mesi.
- Personale sanitario, specializzandi, familiari di degenti: vicinanza all'ospedale, affitti medi/brevi.
- Gli ospedali piccoli generano poca domanda: pesare i poli per posti letto o iscritti.
- Verificare che il prezzo massimo di 120.000 € non escluda quasi tutta l'offerta nelle grandi città.

## File principali
| File | Contenuto |
|---|---|
| `README.md` | questo documento |
| `dati\` | file di partenza (A–E Perplexity, PBF OpenStreetMap, T2 Ministero) |
| `province_iscritti.csv`, `citta.txt`, `filtra_citta.py` | selezione delle province per iscritti fuori regione |
| `poli_ufficiali.csv` | elenco poli con coordinate Nominatim e Google e stato di controllo |
| `tuoi_indirizzi.csv` (nome, ENTE, provincia), `tuoi_indirizzi_geocodificati.csv`, `tuoi_indirizzi_google*.csv` | indirizzi e relativi geocoding |
| `geocoda_google.py` | geocoding con Google Places / Geocoding API |
| `genera_aree.py`, `genera_aree_ufficiali.py` | generazione poligoni e URL Idealista |
| `aree_vendita.csv`, `aree_affitto.csv` | aree di ricerca (152 per operazione) |
| `raccogli_poli.py`, `estrai_poli_pbf.py`, `poli_manuali.csv` | alternativa OpenStreetMap (Overpass / PBF) |
| `estrai_annunci_console.js` | frammento da incollare nella console del browser: copia gli annunci visibili della pagina elenco |
| `leggi_annunci.py` | legge `dati\\pagine\\*.html/.txt` e aggiorna gli annunci |
| `grafici_boxplot.py`, `grafici_boxplot_affitto.py` | boxplot per citta' (prezzo e prezzo al mq; canone e canone al mq) |
| `annunci_vendita.csv`, `annunci_affitto.csv` | annunci raccolti |
| `_to_delete\` | backup e file temporanei, eliminabili a mano |
