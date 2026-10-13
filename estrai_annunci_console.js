// estrai_annunci_console.js
// Da incollare nella CONSOLE del browser (F12 -> Console) sulla pagina ELENCO di Idealista gia' aperta.
// Legge solo gli annunci visibili in quella pagina e li copia negli appunti come testo.
// Non scarica file, non naviga, non fa richieste: usa solo cio' che la pagina ha gia' caricato.
(() => {
  const url = (document.querySelector('link[rel="canonical"]') || {}).href || location.href;
  const annunci = [...document.querySelectorAll('article.item')];
  const pulisci = (a) => {
    const c = a.cloneNode(true);
    c.querySelectorAll('picture,img,svg,script,style,button,noscript,video,source,iframe').forEach(e => e.remove());
    return c.outerHTML.replace(/\s+/g, ' ');
  };
  const testo = '<html><head><link rel="canonical" href="' + url.replace(/&/g, '&amp;') + '"></head><body>\n'
              + annunci.map(pulisci).join('\n') + '\n</body></html>';
  copy(testo);   // funzione della console di Chrome/Edge: copia negli appunti
  console.log('Copiati ' + annunci.length + ' annunci (' + testo.length + ' caratteri). '
            + 'Incolla in un file .txt dentro dati\\pagine (es. FERRARA_affitto.txt).');
})();
