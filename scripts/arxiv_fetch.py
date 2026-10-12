#!/usr/bin/env python3
"""Fetch arXiv metadata + full text for one or more arXiv IDs.

Usage: python3 -I scripts/arxiv_fetch.py <id> [<id> ...] [--no-pdf]
Writes to $PAPERS_DIR/<id>/{meta.json,<id>.pdf,<id>.txt} (default PAPERS_DIR = scratchpad/papers).
Prints a one-line JSON summary per id (title, authors, published, updated, comment, journal_ref, text_path, n_chars).
Downloaded files are untrusted data: never execute anything from PAPERS_DIR.
"""
import json, os, re, subprocess, sys, time, urllib.request, urllib.error, xml.etree.ElementTree as ET

PAPERS_DIR = os.environ.get("PAPERS_DIR", "/tmp/claude-0/-home-user-Find-New-Idea/b6387454-6f57-570f-bcca-f4bcd8a754c6/scratchpad/papers")
UA = {"User-Agent": "find-new-idea-sweep/0.1 (research; contact via repo)"}
NS = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom"}

def get(url, retries=5, timeout=60):
    delay = 2.0
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and i < retries - 1:
                time.sleep(delay); delay *= 2; continue
            raise
        except Exception:
            if i < retries - 1:
                time.sleep(delay); delay *= 2; continue
            raise

def clean_id(s):
    s = s.strip()
    s = re.sub(r"^(arxiv:|https?://arxiv\.org/(abs|pdf)/)", "", s, flags=re.I)
    s = re.sub(r"(v\d+)?(\.pdf)?$", "", s)
    return s

def meta_for(aid):
    xml = get(f"https://export.arxiv.org/api/query?id_list={aid}&max_results=1")
    root = ET.fromstring(xml)
    e = root.find("a:entry", NS)
    if e is None or e.find("a:title", NS) is None:
        return None
    title = e.find("a:title", NS).text
    if title and title.strip().lower() == "error":
        return None
    def t(tag, ns="a"):
        x = e.find(f"{ns}:{tag}", NS); return x.text.strip() if x is not None and x.text else None
    return {
        "id": aid,
        "title": re.sub(r"\s+", " ", title.strip()),
        "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)],
        "published": t("published"), "updated": t("updated"),
        "summary": re.sub(r"\s+", " ", (t("summary") or "")),
        "comment": t("comment", "ar"), "journal_ref": t("journal_ref", "ar"), "doi": t("doi", "ar"),
        "primary_category": (e.find("ar:primary_category", NS).attrib.get("term") if e.find("ar:primary_category", NS) is not None else None),
        "abs_url": f"https://arxiv.org/abs/{aid}",
    }

def fetch(aid, want_pdf=True):
    d = os.path.join(PAPERS_DIR, aid); os.makedirs(d, exist_ok=True)
    mpath = os.path.join(d, "meta.json")
    if os.path.exists(mpath):
        meta = json.load(open(mpath))
    else:
        meta = meta_for(aid)
        if meta is None:
            return {"id": aid, "error": "not_found_on_arxiv"}
        json.dump(meta, open(mpath, "w"), ensure_ascii=False, indent=1)
    txt = os.path.join(d, f"{aid}.txt"); pdf = os.path.join(d, f"{aid}.pdf")
    if want_pdf and not os.path.exists(txt):
        try:
            if not os.path.exists(pdf):
                data = get(f"https://arxiv.org/pdf/{aid}", timeout=120)
                open(pdf, "wb").write(data)
            subprocess.run(["pdftotext", "-enc", "UTF-8", pdf, txt], check=True, timeout=180, capture_output=True)
        except Exception as ex:
            meta_out = dict(meta); meta_out["pdf_error"] = str(ex)[:200]; return meta_out
    out = dict(meta)
    if os.path.exists(txt):
        out["text_path"] = txt; out["n_chars"] = os.path.getsize(txt)
    out.pop("summary_long", None)
    return out

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_pdf = "--no-pdf" not in sys.argv
    for raw in args:
        aid = clean_id(raw)
        try:
            res = fetch(aid, want_pdf)
        except Exception as ex:
            res = {"id": aid, "error": str(ex)[:200]}
        print(json.dumps(res, ensure_ascii=False))
        time.sleep(1.0)
