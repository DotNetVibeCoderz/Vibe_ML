"""Builds the Delta Lake and Iceberg tables DaskNet.Lakehouse is tested against.

The tables are written by the reference implementations (delta-rs and pyiceberg), not by
Gravicode code, so the readers are judged by what real writers produce. Next to each table,
expected/<version>.jsonl holds the rows every version or snapshot must read back as, sorted by id.

    python -m venv venv && venv/Scripts/pip install deltalake "pyiceberg[sql-sqlite,pyarrow]"
    venv/Scripts/python tools/verify/make_lake_fixtures.py

Output: tests/DaskNet.Lakehouse.Tests/Fixtures/{delta,iceberg}/...
"""
import datetime as dt
import json
import os
import shutil
import sys
import tempfile

import pyarrow as pa
import pyarrow.compute as pc

ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'DaskNet.Lakehouse.Tests', 'Fixtures')


def rows(start, count, extra=False):
    """Deterministic rows covering every column type and missing values."""
    ids = list(range(start, start + count))
    cols = {
        'id': pa.array(ids, pa.int64()),
        'region': pa.array([['north', 'south', 'east'][i % 3] for i in ids], pa.string()),
        'amount': pa.array([None if i % 7 == 0 else round(i * 1.25, 2) for i in ids], pa.float64()),
        'name': pa.array([None if i % 11 == 0 else f'item-{i}' for i in ids], pa.string()),
        'flag': pa.array([None if i % 13 == 0 else i % 2 == 0 for i in ids], pa.bool_()),
        'ts': pa.array([dt.datetime(2024, 1, 1) + dt.timedelta(hours=i) for i in ids], pa.timestamp('us')),
    }
    if extra:
        cols['score'] = pa.array([i / 4 for i in ids], pa.float64())
    return pa.table(cols)


def dump(table, path):
    """Rows sorted by id as JSON lines; timestamps as ISO text, floats as written."""
    table = table.sort_by('id')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for row in table.to_pylist():
            out = {}
            for k in sorted(row):
                v = row[k]
                out[k] = v.isoformat() if isinstance(v, (dt.datetime, dt.date)) else v
            f.write(json.dumps(out, sort_keys=True) + '\n')


def delta():
    from deltalake import DeltaTable, write_deltalake

    base = os.path.join(ROOT, 'delta')
    shutil.rmtree(base, ignore_errors=True)

    # sales: partitioned, appended, a predicate delete, a checkpoint, then more appends.
    path = os.path.join(base, 'sales')
    expected = os.path.join(base, 'sales_expected')
    write_deltalake(path, rows(0, 60), partition_by=['region'])                      # v0
    write_deltalake(path, rows(60, 30), partition_by=['region'], mode='append')      # v1
    DeltaTable(path).delete("amount > 90")                                          # v2: rewrites files
    write_deltalake(path, rows(90, 20), partition_by=['region'], mode='append')      # v3
    DeltaTable(path).create_checkpoint()                                             # checkpoint at v3
    write_deltalake(path, rows(110, 15), partition_by=['region'], mode='append')     # v4
    write_deltalake(path, rows(200, 10), partition_by=['region'], mode='overwrite')  # v5
    write_deltalake(path, rows(210, 5), partition_by=['region'], mode='append')      # v6
    table = DeltaTable(path)
    for version in range(table.version() + 1):
        dump(DeltaTable(path, version=version).to_pyarrow_table(), os.path.join(expected, f'{version}.jsonl'))

    # plain: unpartitioned, with schema evolution (a column added by a merge-schema append).
    path = os.path.join(base, 'plain')
    expected = os.path.join(base, 'plain_expected')
    write_deltalake(path, rows(0, 40))
    write_deltalake(path, rows(40, 20, extra=True), mode='append', schema_mode='merge')
    table = DeltaTable(path)
    for version in range(table.version() + 1):
        dump(DeltaTable(path, version=version).to_pyarrow_table(), os.path.join(expected, f'{version}.jsonl'))


def iceberg():
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import DoubleType

    base = os.path.join(ROOT, 'iceberg')
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base)
    catalog_dir = tempfile.mkdtemp()
    try:
        warehouse = os.path.abspath(base).replace('\\', '/')
        catalog = SqlCatalog('fixtures', uri=f'sqlite:///{catalog_dir}/catalog.db', warehouse=warehouse)
        catalog.create_namespace('db')

        def snapshots(table, expected):
            for i, snap in enumerate(table.history()):
                data = table.scan(snapshot_id=snap.snapshot_id).to_arrow()
                dump(data, os.path.join(expected, f'{i}.jsonl'))
            with open(os.path.join(expected, 'snapshots.json'), 'w', encoding='utf-8', newline='\n') as f:
                json.dump([s.snapshot_id for s in table.history()], f)

        # sales: identity-partitioned, appends, a delete (copy-on-write), schema evolution.
        schema = rows(0, 1).schema
        table = catalog.create_table('db.sales', schema=schema)
        with table.update_spec() as spec:
            spec.add_identity('region')
        table.append(rows(0, 60))
        table.append(rows(60, 30))
        table.delete(delete_filter='amount > 90')
        with table.update_schema() as update:
            update.add_column('score', DoubleType())
        table.append(rows(90, 20, extra=True))
        table = catalog.load_table('db.sales')
        snapshots(table, os.path.join(base, 'sales_expected'))

        # renamed: a column renamed between appends; older files still call it 'name'.
        table = catalog.create_table('db.renamed', schema=schema)
        table.append(rows(0, 12))
        with table.update_schema() as update:
            update.rename_column('name', 'label')
        renamed = rows(12, 8).rename_columns(['id', 'region', 'amount', 'label', 'flag', 'ts'])
        table.append(renamed)
        table = catalog.load_table('db.renamed')
        snapshots(table, os.path.join(base, 'renamed_expected'))

        # plain: unpartitioned, overwrite.
        table = catalog.create_table('db.plain', schema=schema)
        table.append(rows(0, 25))
        table.overwrite(rows(100, 10))
        table = catalog.load_table('db.plain')
        snapshots(table, os.path.join(base, 'plain_expected'))
    finally:
        shutil.rmtree(catalog_dir, ignore_errors=True)


if __name__ == '__main__':
    which = sys.argv[1:] or ['delta', 'iceberg']
    if 'delta' in which:
        delta()
    if 'iceberg' in which:
        iceberg()
    print('fixtures written to', os.path.abspath(ROOT))
