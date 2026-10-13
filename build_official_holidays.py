"""Download DGPA office calendars and derive consecutive breaks of >=3 days.

Run once to refresh data/official_holiday_days.csv and
data/official_long_holiday_periods.csv. Requires xlrd for 2007-2016 XLS files.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import re
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import xlrd


ROOT = Path(__file__).resolve().parent
BASE = "https://www.dgpa.gov.tw"
PORTAL = "https://data.gov.tw/dataset/14718"
CHINESE_MONTHS = "一二三四五六七八九十"


def fetch(url: str) -> bytes:
    url = urllib.parse.quote(url, safe="/:?&=%+(),")
    cache = Path("/tmp/dgpa_calendar_cache")
    cache.mkdir(exist_ok=True)
    cached = cache / hashlib.sha256(url.encode()).hexdigest()
    if cached.exists():
        return cached.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    cached.write_bytes(raw)
    return raw


def links(page: bytes) -> list[tuple[str, str]]:
    source = page.decode("utf-8", errors="replace")
    found = []
    for url, label in re.findall(r'<a\b[^>]*href="([^"]+)"[^>]*>(.*?)</a>', source, re.S | re.I):
        label = re.sub(r"<[^>]+>", "", label)
        found.append((urllib.parse.urljoin(BASE, html.unescape(url)), html.unescape(label).strip()))
    return found


def old_calendar_pages() -> dict[int, str]:
    pages = {}
    for page_num in (3, 4):
        for url, label in links(fetch(f"{BASE}/informationlist?page={page_num}&uid=30")):
            match = re.search(r"中華民國([一二三四五六七八九十百零]+)年政府行政機關辦公日曆表", label)
            if match and "圖檔" not in label:
                # Page order is newest first; revised editions, if present, are selected separately.
                year = {"九十六": 2007, "九十七": 2008, "九十八": 2009,
                        "九十九": 2010, "一百": 2011, "一百零一": 2012,
                        "一百零二": 2013, "一百零三": 2014,
                        "一百零四": 2015, "一百零五": 2016}.get(match[1])
                if year is not None and (year not in pages or "修正版" in label):
                    pages[year] = url
    if set(pages) != set(range(2007, 2017)):
        raise ValueError(f"Missing historical DGPA calendar pages: {sorted(set(range(2007, 2017)) - set(pages))}")
    pages[2016] = "https://www.dgpa.gov.tw/information?pid=6180&uid=353"
    return pages


def xls_url(page_url: str) -> str:
    candidates = [(url, label) for url, label in links(fetch(page_url))
                  if label.lower().endswith(".xls") or label.lower().endswith("(xls)")]
    if not candidates:
        raise ValueError(f"No XLS attachment on {page_url}")
    return candidates[0][0]


def parse_old_xls(raw: bytes, year: int) -> dict[date, bool]:
    book = xlrd.open_workbook(file_contents=raw, formatting_info=True)
    sheet = book.sheet_by_index(0)
    # The calendar is printed three months across, with each day represented by
    # two rows: Gregorian day above lunar day. Cell fill 42=work, 45=weekend,
    # 44/46=holiday or substitute day in the oldest DGPA template.
    off_fills = {
        2007: {44, 45, 46, 51}, 2008: {44, 45, 46},
        2009: {13, 15, 45, 46}, 2010: {40, 43, 45, 46},
        2011: {40, 45, 46}, 2012: {40, 45, 50, 52},
        2013: {45}, 2014: {45}, 2015: {45}, 2016: {45},
    }
    work_fills = {
        2007: {11, 42, 43}, 2008: {11, 42},
        2009: {11, 41, 53}, 2010: {11, 41, 52},
        2011: {11, 41}, 2012: {41, 43, 55},
        2013: {64}, 2014: {64}, 2015: {64}, 2016: {9, 64},
    }
    # Palette indices change between workbooks. Check the actual legend so
    # adjusted holidays cannot silently be confused with makeup workdays.
    legend_labels = {
        "星期六、星期日": True, "放假之紀念日及民俗節日": True,
        "放假之節日": True, "放假之紀念日及節日": True,
        "補假": True, "調整放假": True, "調整放假日": True,
        "特定節日": False, "補行上班": False, "補行上班日": False,
        "上班日": False, "放假日": True, "workday": False, "dayoff": True,
    }
    checked = 0
    for row in range(sheet.nrows):
        for col in range(sheet.ncols):
            label = str(sheet.cell_value(row, col)).strip()
            if label not in legend_labels:
                continue
            swatch_col = col - (1 if year <= 2012 else 2)
            fill = book.xf_list[sheet.cell_xf_index(row, swatch_col)].background.pattern_colour_index
            expected_fills = off_fills[year] if legend_labels[label] else work_fills[year]
            if fill not in expected_fills:
                raise ValueError(f"Calendar legend mismatch for {year}: {label} has fill {fill}")
            checked += 1
    if checked < 2:
        raise ValueError(f"Cannot verify work/off legend for {year}")

    month_names = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
                   "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12}
    headers = []
    for row in range(sheet.nrows):
        for col in range(sheet.ncols):
            value = sheet.cell_value(row, col)
            if isinstance(value, str):
                match = re.fullmatch(r"\s*([一二三四五六七八九十]+)\s*月\s*", value)
                if match and match[1] in month_names:
                    headers.append((row, col, month_names[match[1]]))
                elif value.strip() in month_names and col in (3, 10, 17):
                    if str(sheet.cell_value(row, col + 2)).strip() == "月":
                        month_label = value.strip() + str(sheet.cell_value(row, col + 1)).strip()
                        if month_label in month_names:
                            headers.append((row, col - 2, month_names[month_label]))
                elif year == 2016 and value in {"JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                                               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"}:
                    month = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                             "JUL", "AUG", "SEP", "OCT", "NOV", "DEC").index(value) + 1
                    headers.append((row, col - 3, month))
    headers = [(row, col, month) for row, col, month in headers
               if row < 100 and col in (1, 8, 15)]
    if sorted(month for _, _, month in headers) != list(range(1, 13)):
        raise ValueError(f"Cannot locate twelve month grids for {year}: {headers}")
    days = {}
    for start_row, start_col, month in headers:
        row_step = 1 if year == 2016 else 2
        for row in range(start_row + 2, min(start_row + (8 if year == 2016 else 14), sheet.nrows), row_step):
            for col in range(start_col, min(start_col + 7, sheet.ncols)):
                value = sheet.cell_value(row, col)
                if isinstance(value, (int, float)) and 1 <= value <= 31 and value == int(value):
                    try:
                        current = date(year, month, int(value))
                    except ValueError:
                        continue
                    fill = book.xf_list[sheet.cell_xf_index(row, col)].background.pattern_colour_index
                    if fill not in off_fills[year] | work_fills[year]:
                        raise ValueError(f"Unknown fill {fill} on {current} in {year}")
                    days[current] = fill in off_fills[year]
    expected = {date(year, 1, 1) + timedelta(days=offset)
                for offset in range((date(year + 1, 1, 1) - date(year, 1, 1)).days)}
    if set(days) != expected:
        raise ValueError(f"Incomplete XLS calendar for {year}: {len(days)} days; missing {sorted(expected - set(days))[:5]}")
    return days


def new_csv_urls() -> dict[int, str]:
    urls = {}
    source = fetch(PORTAL).decode("utf-8")
    resources = re.findall(
        r'<li class="resource-item".*?<a href="([^"]+)".*?<span data-v-[^>]+>([^<]+)</span>',
        source, re.S,
    )
    for raw_url, label in resources:
        url = html.unescape(raw_url)
        match = re.search(r"(10[6-9]|11[0-5])年.*政府行政機關辦公日曆表", label)
        if match and "Google" not in label:
            urls[int(match[1]) + 1911] = url
    if set(urls) != set(range(2017, 2027)):
        raise ValueError(f"Missing CSV years: {sorted(set(range(2017, 2027)) - set(urls))}")
    return urls


def parse_new_csv(raw: bytes, year: int) -> dict[date, bool]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp950")
    rows = csv.DictReader(io.StringIO(text))
    if not {"西元日期", "是否放假"}.issubset(rows.fieldnames or []):
        raise ValueError(f"Unexpected official CSV columns for {year}: {rows.fieldnames}")
    days = {}
    for row in rows:
        current = date.fromisoformat(row["西元日期"].strip().replace("/", "-"))
        if current.year != year or row["是否放假"].strip() not in {"0", "2"}:
            raise ValueError(f"Unexpected official CSV record: {row}")
        days[current] = row["是否放假"].strip() == "2"
    expected = (date(year + 1, 1, 1) - date(year, 1, 1)).days
    if len(days) != expected:
        raise ValueError(f"Incomplete official CSV for {year}: {len(days)} days")
    return days


def main() -> None:
    days, sources = {}, {}
    # The analysis starts in January 2007. Include the two preceding official
    # weekend days so the Dec 30–Jan 1 break is recognized across the boundary.
    boundary_page = f"{BASE}/information?pid=4971&uid=30"
    boundary_book = xlrd.open_workbook(file_contents=fetch(xls_url(boundary_page)), formatting_info=True)
    boundary_sheet = boundary_book.sheet_by_index(0)
    for current, row, col in [(date(2006, 12, 30), 54, 21),
                              (date(2006, 12, 31), 56, 15)]:
        fill = boundary_book.xf_list[boundary_sheet.cell_xf_index(row, col)].background.pattern_colour_index
        if boundary_sheet.cell_value(row, col) != current.day or fill != 45:
            raise ValueError(f"Unexpected 2006 boundary calendar cell for {current}")
        days[current] = True
    sources[2006] = boundary_page
    for year, page_url in sorted(old_calendar_pages().items()):
        url = xls_url(page_url)
        parsed = parse_old_xls(fetch(url), year)
        days.update(parsed)
        sources[year] = page_url
        print(year, len(parsed), "XLS", url)
    for year, url in sorted(new_csv_urls().items()):
        parsed = parse_new_csv(fetch(url), year)
        days.update(parsed)
        sources[year] = url
        print(year, len(parsed), "CSV", url)
    output = ROOT / "data"
    with (output / "official_holiday_days.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["date", "is_day_off", "source_url"])
        for current, day_off in sorted(days.items()):
            writer.writerow([current.isoformat(), int(day_off), sources[current.year]])
    periods = []
    start = previous = None
    for current, day_off in sorted(days.items()):
        if day_off:
            if previous is None:
                start = current
            previous = current
        elif previous is not None:
            if (previous - start).days + 1 >= 3:
                periods.append((start, previous))
            start = previous = None
    if previous is not None and (previous - start).days + 1 >= 3:
        periods.append((start, previous))
    with (output / "official_long_holiday_periods.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["start_date", "end_date", "duration_days"])
        for start, end in periods:
            writer.writerow([start.isoformat(), end.isoformat(), (end - start).days + 1])
    print(f"Wrote {len(days)} calendar days and {len(periods)} long holidays")


if __name__ == "__main__":
    main()
