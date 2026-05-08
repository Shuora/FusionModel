                      

from __future__ import annotations

import csv

import re

import urllib.parse

import urllib.request

from collections import defaultdict

from dataclasses import dataclass

from datetime import date

from html.parser import HTMLParser

from pathlib import Path



BASE = "https://www.malware-traffic-analysis.net"

YEARS = (2020, 2021, 2022, 2023)

START = date(2020, 2, 1)

END = date(2023, 2, 28)

UA = "Mozilla/5.0"



FAMILY_PATTERNS = {

    "Dridex": re.compile(r"\\bdridex\\b", re.I),

    "Emotet": re.compile(r"\\bemotet\\b", re.I),

    "Hancitor": re.compile(r"\\bhancitor\\b", re.I),

    "IcedID": re.compile(r"\\biced[\\s-]?id\\b", re.I),

    "Qakbot": re.compile(r"\\bqakbot\\b|\\bqbot\\b", re.I),

    "Trickbot": re.compile(r"\\btrickbot\\b", re.I),

    "Ursnif": re.compile(r"\\bursnif\\b|\\bgozi\\b|\\bisfb\\b", re.I),

}

DAY_PAGE_RE = re.compile(r"/(\\d{4})/(\\d{2})/(\\d{2})/index\\d*\\.html$")

ZIP_RE = re.compile(r"pcap\\.zip", re.I)



class LinkParser(HTMLParser):

    def __init__(self):

        super().__init__()

        self.links = []

    def handle_starttag(self, tag, attrs):

        if tag.lower() != "a":

            return

        for k, v in attrs:

            if k and k.lower() == "href" and v:

                self.links.append(v.strip())



@dataclass(frozen=True)

class CandidateRow:

    dt: date

    family: str

    page_url: str

    pcap_zip_url: str





def fetch(url: str, timeout: int = 30) -> str:

    req = urllib.request.Request(url, headers={"User-Agent": UA})

    with urllib.request.urlopen(req, timeout=timeout) as resp:

        return resp.read().decode("utf-8", "ignore")





def extract_links(html: str):

    p = LinkParser()

    p.feed(html)

    return p.links





def to_abs(base_url: str, href: str) -> str:

    return urllib.parse.urljoin(base_url, href)





def iter_day_pages(index_url: str, html: str):

    out = set()

    for href in extract_links(html):

        u = to_abs(index_url, href)

        m = DAY_PAGE_RE.search(u)

        if not m:

            continue

        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

        if START <= d <= END:

            out.add((d, u))

    return sorted(out)





def detect_families(page_text: str):

    found = []

    for fam, pat in FAMILY_PATTERNS.items():

        if pat.search(page_text):

            found.append(fam)

    return found





def collect_rows():

    rows = set()

    idx_err = []

    page_err = []

    for year in YEARS:

        idx = f"{BASE}/{year}/index.html"

        try:

            idx_html = fetch(idx)

        except Exception as e:

            idx_err.append((idx, type(e).__name__))

            continue



        for d, day_page in iter_day_pages(idx, idx_html):

            try:

                day_html = fetch(day_page)

            except Exception as e:

                page_err.append((day_page, type(e).__name__))

                continue



            families = detect_families(day_html)

            if not families:

                continue



            zip_links = [to_abs(day_page, href) for href in extract_links(day_html) if ZIP_RE.search(href)]

            if not zip_links:

                continue



            for fam in families:

                for z in zip_links:

                    rows.add(CandidateRow(d, fam, day_page, z))



    return sorted(rows, key=lambda x: (x.dt, x.family, x.page_url, x.pcap_zip_url)), idx_err, page_err





def main():

    out_tsv = Path("/tmp/mta_candidates_202002_202302.tsv")

    out_dir = Path("/tmp/mta_candidates_by_family")

    out_dir.mkdir(parents=True, exist_ok=True)



    rows, idx_err, page_err = collect_rows()



    with out_tsv.open("w", newline="", encoding="utf-8") as f:

        w = csv.writer(f, delimiter="\t")

        w.writerow(["date", "family", "page_url", "pcap_zip_url"])

        for r in rows:

            w.writerow([r.dt.isoformat(), r.family, r.page_url, r.pcap_zip_url])



    by_family = defaultdict(list)

    for r in rows:

        by_family[r.family].append(r)



    for fam, items in sorted(by_family.items()):

        p = out_dir / f"{fam}.txt"

        with p.open("w", encoding="utf-8") as f:

            for r in items:

                f.write(f"{r.dt.isoformat()}\t{r.page_url}\t{r.pcap_zip_url}\n")



    print(f"rows_total={len(rows)}")

    for fam in sorted(by_family):

        uniq_pages = len({x.page_url for x in by_family[fam]})

        uniq_zips = len({x.pcap_zip_url for x in by_family[fam]})

        print(f"{fam}\trows={len(by_family[fam])}\tpages={uniq_pages}\tzips={uniq_zips}")



    print(f"index_errors={len(idx_err)} page_errors={len(page_err)}")

    if idx_err:

        print("index_error_examples:")

        for u, e in idx_err[:10]:

            print(f"{u}\t{e}")

    if page_err:

        print("page_error_examples:")

        for u, e in page_err[:10]:

            print(f"{u}\t{e}")



    print(f"saved_tsv={out_tsv}")

    print(f"saved_family_dir={out_dir}")



if __name__ == "__main__":

    main()

