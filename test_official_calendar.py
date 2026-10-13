"""Regression checks for DGPA daily and long-holiday exports."""
import csv
import unittest
from datetime import date, timedelta

from build_official_holidays import (EXPECTED_DAY_OFF_COUNTS, ROOT,
                                     classify_calendar_day)


class OfficialCalendarTests(unittest.TestCase):
    def test_special_observance_preserves_weekend(self):
        off, work, special = {45}, {11, 41}, {11}
        self.assertTrue(classify_calendar_day(11, date(2010, 5, 1), off, work, special))
        self.assertTrue(classify_calendar_day(11, date(2011, 5, 1), off, work, special))
        self.assertTrue(classify_calendar_day(11, date(2011, 9, 3), off, work, special))
        self.assertFalse(classify_calendar_day(11, date(2009, 5, 1), off, work, special))
        self.assertFalse(classify_calendar_day(41, date(2011, 5, 2), off, work, special))

    def test_exports_cover_all_dates_and_match_official_annual_totals(self):
        with (ROOT / 'data/official_holiday_days.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        actual = {date.fromisoformat(row['date']): int(row['is_day_off']) for row in rows}
        self.assertEqual(len(actual), len(rows))
        self.assertEqual(set(actual), {date(2006, 12, 30) + timedelta(days=i)
                                      for i in range(len(rows))})
        for year, expected in EXPECTED_DAY_OFF_COUNTS.items():
            self.assertEqual(sum(value for day, value in actual.items() if day.year == year),
                             expected, year)
        for day in (date(2010, 5, 1), date(2011, 5, 1), date(2011, 9, 3)):
            self.assertEqual(actual[day], 1, day)

    def test_periods_are_exact_maximal_runs_and_key_years_match_announcements(self):
        with (ROOT / 'data/official_holiday_days.csv').open(newline='') as stream:
            actual = {date.fromisoformat(row['date']): bool(int(row['is_day_off']))
                      for row in csv.DictReader(stream)}
        expected_periods = []
        start = None
        for day, off in sorted(actual.items()):
            if off and start is None:
                start = day
            if not off and start is not None:
                end = day - timedelta(days=1)
                if (end - start).days >= 2:
                    expected_periods.append((start.isoformat(), end.isoformat(),
                                             (end - start).days + 1))
                start = None
        if start is not None:
            end = max(actual)
            if (end - start).days >= 2:
                expected_periods.append((start.isoformat(), end.isoformat(),
                                         (end - start).days + 1))
        with (ROOT / 'data/official_long_holiday_periods.csv').open(newline='') as stream:
            periods = [(row['start_date'], row['end_date'], int(row['duration_days']))
                       for row in csv.DictReader(stream)]
        self.assertEqual(periods, expected_periods)
        self.assertEqual([p for p in periods if p[0].startswith('2008-')], [
            ('2008-02-06', '2008-02-11', 6),
            ('2008-04-04', '2008-04-06', 3),
            ('2008-10-10', '2008-10-12', 3)])
        self.assertEqual([p for p in periods if p[0].startswith('2009-')], [
            ('2009-01-01', '2009-01-04', 4),
            ('2009-01-24', '2009-02-01', 9),
            ('2009-05-28', '2009-05-31', 4)])


if __name__ == '__main__':
    unittest.main()
