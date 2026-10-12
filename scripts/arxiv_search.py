#!/usr/bin/env python3
"""Search arXiv via the export API.

Usage: python3 -I scripts/arxiv_search.py "<query>" [--max N] [--from YYYYMMDD] [--to YYYYMMDD] [--sort relevance|date]
Query syntax is arXiv API syntax, e.g.  'all:"kv cache" AND all:quantization'  or  'ti:quantization AND abs:softmax'.
Prints TSV: id <TAB> published <TAB> primary_cat <TAB> title <TAB> first_author
"""
import re, sys, time, urllib.parse, urllib.request, urllib.error, xml.etree.ElementTree as ET
NS = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom"}
UA = {"User-Agent": "find-new-idea-sweep/0.1"}

def main():
    argv = sys.argv[1:]
    q = argv[0]; maxn = 50; dfrom = None; dto = None; sort = "relevance"
    i = 1
    while i < len(argv):
        if argv[i] == "--max": maxn = int(argv[i+1]); i += 2
        elif argv[i] == "--from": dfrom = argv[i+1]; i += 2
        elif argv[i] == "--to": dto = argv[i+1]; i += 2
        elif argv[i] == "--sort": sort = argv[i+1]; i += 2
        else: i += 1
    if dfrom or dto:
        q = f"({q}) AND submittedDate:[{(dfrom or '19910101')}0000 TO {(dto or '20991231')}2359]"
    sortBy = "submittedDate" if sort == "date" else "relevance"
    got = 0; start = 0
    while got < maxn:
        n = min(100, maxn - got)
        url = ("https://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q) +
               f"&start={start}&max_results={n}&sortBy={sortBy}&sortOrder=descending")
        delay = 3.0
        for attempt in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                    xml = r.read(); break
            except Exception:
                if attempt == 4: raise
                time.sleep(delay); delay *= 2
        root = ET.fromstring(xml)
        entries = root.findall("a:entry", NS)
        if not entries: break
        for e in entries:
            aid = e.find("a:id", NS).text.rsplit("/", 1)[-1]
            aid = re.sub(r"v\d+$", "", aid)
            title = re.sub(r"\s+", " ", e.find("a:title", NS).text.strip())
            pub = (e.find("a:published", NS).text or "")[:10]
            cat = e.find("ar:primary_category", NS)
            cat = cat.attrib.get("term") if cat is not None else ""
            au = e.find("a:author/a:name", NS)
            print("\t".join([aid, pub, cat, title, au.text if au is not None else ""]))
            got += 1
        start += len(entries)
        if len(entries) < n: break
        time.sleep(3.0)

if __name__ == "__main__":
    main()
