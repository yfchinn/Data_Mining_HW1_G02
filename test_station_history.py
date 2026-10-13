"""Run with: python -m unittest test_station_history"""
import unittest

import pandas as pd

from model_data import load_model_data, make_time_series_cv
from station_history import operating_target, features_for_target, fold_info, common_windows


class OperatingHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_model_data()
        cls.calendar = pd.concat([cls.data.X_train, cls.data.X_test])
        cls.targets = pd.concat([cls.data.y_train, cls.data.y_test])

    def test_mask_keeps_real_zeros_and_monthly_alignment(self):
        y = self.targets.hsr_entry_nangang.copy()
        y.loc['2016-08-01'] = 0
        masked = operating_target(y)
        self.assertTrue(masked.loc[:'2016-06-01'].isna().all())
        self.assertEqual(masked.loc['2016-08-01'], 0)
        pd.testing.assert_index_equal(masked.index, y.index)
        X = features_for_target(self.calendar, y)
        self.assertEqual(X.dropna().index[0], pd.Timestamp('2017-07-01'))
        self.assertEqual(X.loc['2017-08-01', 'lag_12'], 0)

    def test_no_preopening_or_future_values_enter_features(self):
        y = self.targets.hsr_entry_nangang.copy()
        expected = features_for_target(self.calendar, y)
        y.loc[:'2016-06-01'] = 999999
        pd.testing.assert_frame_equal(expected, features_for_target(self.calendar, y))
        y.loc['2024-07-01':] += 999999
        actual = features_for_target(self.calendar, y)
        pd.testing.assert_frame_equal(expected.loc[:'2024-07-01'], actual.loc[:'2024-07-01'])

    def test_first_usable_months_and_skipped_folds(self):
        first = {'nangang': '2017-07-01', 'miaoli': '2016-12-01',
                 'changhua': '2016-12-01', 'yunlin': '2016-12-01',
                 'taipei': '2008-03-01', 'taichung': '2008-01-01'}
        folds = list(make_time_series_cv(self.data).split(self.data.X_train))
        for station, month in first.items():
            for direction in ('entry', 'exit'):
                target = f'hsr_{direction}_{station}'
                X = features_for_target(self.calendar, self.targets[target])
                self.assertEqual(X.dropna().index[0], pd.Timestamp(month))
                statuses = []
                for i, (tr, va) in enumerate(folds, 1):
                    train = X.loc[self.data.X_train.index[tr]].dropna()
                    val = X.loc[self.data.X_train.index[va]]
                    statuses.append(fold_info(target, i, train, val)['status'])
                self.assertEqual(statuses, ['skipped', 'skipped', 'used', 'used']
                                 if station in ('nangang', 'miaoli', 'changhua', 'yunlin')
                                 else ['used'] * 4)

    def test_common_windows_exclude_unavailable_120_months(self):
        cache = {t: features_for_target(self.calendar, self.targets[t]) for t in self.targets}
        windows, audit = common_windows(cache, self.data.X_train.index[-24:], (36, 60, 120))
        self.assertEqual(windows, (36, 60))
        row = audit[(audit.target == 'hsr_entry_nangang') & (audit.window_months == 120)].iloc[0]
        self.assertEqual(row.min_available_train, 60)
        self.assertFalse(row.eligible)


if __name__ == '__main__':
    unittest.main()
