"""Chunked optional Home Credit history features, without target or ID predictors.

Means are total nonmissing sums / total nonmissing counts across all chunks.
Applicant IDs only provide the join index. No optional table is required.
"""
from pathlib import Path
import numpy as np
import pandas as pd


_TABLES = {
    'previous_application': 'PREVIOUS',
    'installments_payments': 'INSTALLMENTS',
    'credit_card_balance': 'CREDIT_CARD',
    'POS_CASH_balance': 'POS_CASH',
}


def _find_table(directory, stem):
    canonical = directory / (stem + '.csv')
    if canonical.exists():
        return canonical
    # Browser downloads sometimes append a parenthesized copy number.
    matches = sorted(directory.glob(stem + '(*).csv'))
    if len(matches) > 1:
        raise ValueError(f'Ambiguous copies of {stem}: {matches}')
    return matches[0] if matches else None


def _merge_extreme(left, right, maximum=False):
    if left is None:
        return right
    # fmin/fmax keep the observed value when only one side is missing.
    left, right = left.align(right, join='outer')
    values = np.fmax(left.to_numpy(), right.to_numpy()) if maximum else np.fmin(left.to_numpy(), right.to_numpy())
    return pd.DataFrame(values, index=left.index, columns=left.columns)


def _aggregate(path, prefix, chunksize):
    sums = counts = minima = maxima = row_counts = None
    rows = missing_ids = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        rows += len(chunk)
        if 'SK_ID_CURR' not in chunk:
            raise ValueError(f'{path.name} has no SK_ID_CURR')
        missing_ids += int(chunk['SK_ID_CURR'].isna().sum())
        chunk = chunk.dropna(subset=['SK_ID_CURR']).copy()
        if chunk.empty:
            continue
        if prefix == 'PREVIOUS':
            days = [c for c in chunk if c.startswith('DAYS_')]
            chunk[days] = chunk[days].replace(365243, np.nan)
        if prefix == 'INSTALLMENTS':
            if {'AMT_PAYMENT', 'AMT_INSTALMENT'}.issubset(chunk):
                chunk['PAYMENT_TO_DUE_RATIO'] = chunk['AMT_PAYMENT'] / chunk['AMT_INSTALMENT'].replace(0, np.nan)
            if {'DAYS_ENTRY_PAYMENT', 'DAYS_INSTALMENT'}.issubset(chunk):
                chunk['PAYMENT_DAYS_MINUS_DUE'] = chunk['DAYS_ENTRY_PAYMENT'] - chunk['DAYS_INSTALMENT']
        # Every SK_ID_* column is an identifier, never a numeric predictor.
        columns = [c for c in chunk.select_dtypes(include='number')
                   if not c.startswith('SK_ID_') and c != 'TARGET']
        chunk[columns] = chunk[columns].replace([np.inf, -np.inf], np.nan)
        groups = chunk.groupby('SK_ID_CURR', sort=False)
        sizes = groups.size()
        row_counts = sizes if row_counts is None else row_counts.add(sizes, fill_value=0)
        if not columns:
            continue
        grouped = groups[columns]
        part_sum = grouped.sum(min_count=1)
        part_count = grouped.count()
        sums = part_sum if sums is None else sums.add(part_sum, fill_value=0)
        counts = part_count if counts is None else counts.add(part_count, fill_value=0)
        minima = _merge_extreme(minima, grouped.min())
        maxima = _merge_extreme(maxima, grouped.max(), maximum=True)
    if row_counts is None:
        result = pd.DataFrame(index=pd.Index([], name='SK_ID_CURR'))
    else:
        result = row_counts.rename(f'{prefix}_ROW_COUNT').to_frame()
        if sums is not None:
            counts = counts.fillna(0)
            sums = sums.where(counts.gt(0))
            means = sums / counts.replace(0, np.nan)
            for stat, frame in [('SUM', sums), ('COUNT', counts), ('MIN', minima), ('MAX', maxima), ('MEAN', means)]:
                features = frame.rename(columns=lambda c: f'{prefix}_{c}_{stat}')
                result = result.join(features)
    return result, rows, missing_ids


def build_extra_history(data_dir, chunksize=500000):
    """Return one numeric feature row per applicant in optional history CSVs.

    ``attrs['row_counts']`` records source rows for each found table;
    ``attrs['missing_id_rows']`` records excluded source rows with missing IDs.
    A missing table is skipped; no history is represented by NaN on a later
    left join to application data, distinct from an observed zero balance.
    """
    if not isinstance(chunksize, int) or not 1 <= chunksize <= 500000:
        raise ValueError('chunksize must be an integer between 1 and 500000')
    directory = Path(data_dir)
    frames, row_counts, missing_id_rows = [], {}, {}
    for stem, prefix in _TABLES.items():
        path = _find_table(directory, stem)
        if path is None:
            continue
        frame, rows, missing_ids = _aggregate(path, prefix, chunksize)
        frames.append(frame)
        row_counts[stem] = rows
        missing_id_rows[stem] = missing_ids
    result = pd.concat(frames, axis=1, join='outer') if frames else pd.DataFrame(index=pd.Index([], name='SK_ID_CURR'))
    result.index.name = 'SK_ID_CURR'
    if not result.index.is_unique:
        raise AssertionError('History aggregation produced duplicate applicants')
    result.attrs['row_counts'] = row_counts
    result.attrs['missing_id_rows'] = missing_id_rows
    for name, count in row_counts.items():
        result.attrs[name + '_rows'] = count
    return result
