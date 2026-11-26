"""Builds the Spark-written lake tables DaskNet.Lakehouse is tested against: Delta deletion vectors,
Delta column mapping with renames and drops, and Apache Hudi copy-on-write / merge-on-read tables.

delta-rs (make_lake_fixtures.py) cannot produce these: its DELETE rewrites files instead of
writing deletion vectors, it cannot rename or drop a mapped column, and it does not write Hudi. So
these come from the reference writers themselves - Spark 3.5 with Delta Lake 3.3 and Hudi 1.0.
Next to each table, <table>_expected/<n>.jsonl holds what Spark reads back for version (or
instant) n, sorted by id.

    python -m venv sparkvenv
    sparkvenv/Scripts/pip install pyspark==3.5.5 delta-spark==3.3.2
    # Windows only: Hadoop needs winutils.exe and hadoop.dll (e.g. github.com/cdarlint/winutils,
    # hadoop-3.3.6/bin) in a folder named by HADOOP_HOME (default: ./hadoop next to this script)
    sparkvenv/Scripts/python tools/verify/make_lake_fixtures_spark.py [delta] [hudi]

Java 17 or 21 must be on PATH; the Delta and Hudi jars are fetched from Maven Central.
Output: tests/DaskNet.Lakehouse.Tests/Fixtures/{delta,hudi}/...
"""
import json
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent / 'tests' / 'DaskNet.Lakehouse.Tests' / 'Fixtures'


def session():
    hadoop = os.environ.get('HADOOP_HOME') or str(pathlib.Path(__file__).resolve().parent / 'hadoop')
    os.environ['HADOOP_HOME'] = hadoop
    os.environ['PATH'] = os.path.join(hadoop, 'bin') + os.pathsep + os.environ['PATH']
    os.environ['PYSPARK_PYTHON'] = sys.executable
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master('local[1]')
             .config('spark.jars.packages', 'io.delta:delta-spark_2.12:3.3.2,org.apache.hudi:hudi-spark3.5-bundle_2.12:1.0.2')
             .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension,org.apache.spark.sql.hudi.HoodieSparkSessionExtension')
             .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog')
             .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')
             .config('spark.sql.session.timeZone', 'UTC')
             .config('spark.sql.shuffle.partitions', '1')
             .config('spark.ui.enabled', 'false')
             .getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    return spark


def rows(spark, start, count, region='region', name='name', seq=None):
    """Deterministic rows, built in SQL (Python workers crash on Windows), with missing values.
    Same shape as make_lake_fixtures.py's rows()."""
    seq_col = f', CAST({seq} AS BIGINT) AS seq' if seq is not None else ''
    return spark.sql(f"""
        SELECT id,
               CASE id % 3 WHEN 0 THEN 'north' WHEN 1 THEN 'south' ELSE 'east' END AS {region},
               CASE WHEN id % 7 = 0 THEN NULL ELSE CAST(ROUND(id * 1.25, 2) AS DOUBLE) END AS amount,
               CASE WHEN id % 11 = 0 THEN NULL ELSE CONCAT('item-', id) END AS {name},
               CASE WHEN id % 13 = 0 THEN NULL ELSE id % 2 = 0 END AS flag,
               TIMESTAMP '2024-01-01 00:00:00' + make_interval(0, 0, 0, 0, CAST(id AS INT), 0, 0) AS ts
               {seq_col}
        FROM range({start}, {start + count})""")


def dump(frame, path, drop=()):
    """Rows sorted by id as JSON lines with every column present; timestamps as UTC ISO text."""
    columns = [c for c in frame.columns if c not in drop]
    ts = [c for c, t in frame.dtypes if t == 'timestamp' and c in columns]
    frame = frame.selectExpr(*[f"date_format({c}, \"yyyy-MM-dd'T'HH:mm:ss\") AS {c}" if c in ts else f'`{c}`' for c in columns])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for row in sorted(frame.collect(), key=lambda r: r['id']):
            d = row.asDict()
            f.write(json.dumps({k: d[k] for k in sorted(d)}, sort_keys=True) + '\n')


def tidy(path):
    """Hadoop's .crc side files are not part of either format; they only add bulk."""
    for folder, _, files in os.walk(path):
        for f in files:
            if f.startswith('.') and f.endswith('.crc'):
                os.remove(os.path.join(folder, f))


def delta(spark):
    from delta.tables import DeltaTable
    base = ROOT / 'delta'

    # dv: partitioned, deletion vectors on, a checkpoint every 3 commits so one carries DVs.
    path = base / 'dv'
    shutil.rmtree(path, ignore_errors=True)
    p = path.as_posix()
    props = "TBLPROPERTIES ('delta.enableDeletionVectors' = 'true', 'delta.checkpointInterval' = '3')"
    rows(spark, 0, 60).coalesce(1).createOrReplaceTempView('src')
    spark.sql(f"CREATE TABLE delta.`{p}` USING delta PARTITIONED BY (region) {props} AS SELECT * FROM src")   # v0
    spark.sql(f"DELETE FROM delta.`{p}` WHERE id % 5 = 0")                                                  # v1: DVs
    spark.sql(f"DELETE FROM delta.`{p}` WHERE amount > 60")                                                 # v2: DVs grow
    rows(spark, 60, 20).coalesce(1).write.format('delta').mode('append').save(p)                            # v3 + checkpoint
    spark.sql(f"DELETE FROM delta.`{p}` WHERE id IN (61, 62, 63, 1)")                                        # v4
    spark.sql(f"UPDATE delta.`{p}` SET amount = -1 WHERE id = 7")                                            # v5: DV + new file
    latest = DeltaTable.forPath(spark, p).history().agg({'version': 'max'}).collect()[0][0]
    for v in range(latest + 1):
        dump(spark.read.format('delta').option('versionAsOf', v).load(p), str(base / 'dv_expected' / f'{v}.jsonl'))
    tidy(path)

    # mapped: column mapping by name; a column renamed, the partition column renamed, one dropped.
    path = base / 'mapped'
    shutil.rmtree(path, ignore_errors=True)
    p = path.as_posix()
    rows(spark, 0, 30).coalesce(1).createOrReplaceTempView('src')
    spark.sql(f"""CREATE TABLE delta.`{p}` USING delta PARTITIONED BY (region)
                  TBLPROPERTIES ('delta.columnMapping.mode' = 'name', 'delta.minReaderVersion' = '2', 'delta.minWriterVersion' = '5')
                  AS SELECT * FROM src""")                                                                  # v0
    spark.sql(f"ALTER TABLE delta.`{p}` RENAME COLUMN name TO label")                                       # v1
    rows(spark, 30, 10, name='label').coalesce(1).write.format('delta').mode('append').save(p)              # v2
    spark.sql(f"ALTER TABLE delta.`{p}` RENAME COLUMN region TO area")                                      # v3
    spark.sql(f"ALTER TABLE delta.`{p}` DROP COLUMN flag")                                                  # v4
    rows(spark, 40, 10, region='area', name='label').drop('flag').coalesce(1).write.format('delta').mode('append').save(p)  # v5
    latest = DeltaTable.forPath(spark, p).history().agg({'version': 'max'}).collect()[0][0]
    for v in range(latest + 1):
        dump(spark.read.format('delta').option('versionAsOf', v).load(p), str(base / 'mapped_expected' / f'{v}.jsonl'))
    tidy(path)

    # mapped_id: column mapping by id, with a rename.
    path = base / 'mapped_id'
    shutil.rmtree(path, ignore_errors=True)
    p = path.as_posix()
    rows(spark, 0, 12).coalesce(1).createOrReplaceTempView('src')
    spark.sql(f"""CREATE TABLE delta.`{p}` USING delta
                  TBLPROPERTIES ('delta.columnMapping.mode' = 'id', 'delta.minReaderVersion' = '2', 'delta.minWriterVersion' = '5')
                  AS SELECT * FROM src""")                                                                  # v0
    spark.sql(f"ALTER TABLE delta.`{p}` RENAME COLUMN amount TO price")                                     # v1
    for v in range(2):
        dump(spark.read.format('delta').option('versionAsOf', v).load(p), str(base / 'mapped_id_expected' / f'{v}.jsonl'))
    tidy(path)


HUDI_META = ('_hoodie_commit_time', '_hoodie_commit_seqno', '_hoodie_record_key', '_hoodie_partition_path', '_hoodie_file_name')


def hudi(spark):
    base = ROOT / 'hudi'
    shutil.rmtree(base, ignore_errors=True)

    def options(name, kind, partitioned=True):
        o = {
            'hoodie.table.name': name,
            'hoodie.datasource.write.recordkey.field': 'id',
            'hoodie.datasource.write.precombine.field': 'seq',
            'hoodie.datasource.write.table.type': kind,
            'hoodie.datasource.write.hive_style_partitioning': 'true',
            'hoodie.metadata.enable': 'false',
            # The default bloom filter is sized for 60,000 keys and makes every file ~430 KB.
            'hoodie.index.bloom.num_entries': '100',
            'hoodie.bloom.index.filter.type': 'SIMPLE',
            'hoodie.parquet.small.file.limit': '0',
            'hoodie.upsert.shuffle.parallelism': '1',
            'hoodie.insert.shuffle.parallelism': '1',
            'hoodie.delete.shuffle.parallelism': '1',
        }
        if partitioned:
            o['hoodie.datasource.write.partitionpath.field'] = 'region'
        else:
            o['hoodie.datasource.write.keygenerator.class'] = 'org.apache.hudi.keygen.NonpartitionedKeyGenerator'
            o['hoodie.datasource.write.partitionpath.field'] = ''
        return o

    def write(frame, p, opts, operation, mode='append'):
        frame.coalesce(1).write.format('hudi').options(**opts).option('hoodie.datasource.write.operation', operation).mode(mode).save(p)

    def instants(local):
        timeline = local / '.hoodie' / 'timeline'
        names = os.listdir(timeline) if timeline.is_dir() else os.listdir(local / '.hoodie')
        done = sorted(n.split('.')[0].split('_')[0] for n in names
                      if n.split('.')[-1] in ('commit', 'deltacommit', 'replacecommit') and not n.startswith('.'))
        return done

    def record(local, uri, query='snapshot'):
        expected = base / (local.name + '_expected')
        done = instants(local)
        for i, instant in enumerate(done):
            reader = spark.read.format('hudi').option('as.of.instant', instant)
            if query != 'snapshot':
                reader = reader.option('hoodie.datasource.query.type', query)
            dump(reader.load(uri), str(expected / f'{i}.jsonl'), drop=HUDI_META)
        with open(expected / 'instants.json', 'w', encoding='utf-8', newline='\n') as f:
            json.dump(done, f)
        tidy(local)

    # cow: copy-on-write, partitioned; insert, upsert, delete, then insert_overwrite (a replacecommit).
    local = base / 'cow'
    uri = 'file:///' + local.as_posix()
    o = options('cow', 'COPY_ON_WRITE')
    write(rows(spark, 0, 30, seq=1), uri, o, 'insert', mode='overwrite')
    write(rows(spark, 25, 10, seq=2).selectExpr('id', 'region', 'amount + 1000 AS amount', 'name', 'flag', 'ts', 'seq'), uri, o, 'upsert')
    write(rows(spark, 0, 30, seq=3).where('id IN (3, 4, 5)'), uri, o, 'delete')
    write(rows(spark, 100, 6, seq=4).where("region = 'north'"), uri, o, 'insert_overwrite')
    record(local, uri)

    # mor: merge-on-read; the upsert lands in log files, so only read-optimized reads are possible.
    local = base / 'mor'
    uri = 'file:///' + local.as_posix()
    o = options('mor', 'MERGE_ON_READ')
    write(rows(spark, 0, 20, seq=1), uri, o, 'insert', mode='overwrite')
    write(rows(spark, 15, 10, seq=2).selectExpr('id', 'region', 'amount + 1000 AS amount', 'name', 'flag', 'ts', 'seq'), uri, o, 'upsert')
    record(local, uri, query='read_optimized')

    # flat: copy-on-write, not partitioned.
    local = base / 'flat'
    uri = 'file:///' + local.as_posix()
    o = options('flat', 'COPY_ON_WRITE', partitioned=False)
    write(rows(spark, 0, 12, seq=1), uri, o, 'insert', mode='overwrite')
    write(rows(spark, 10, 5, seq=2), uri, o, 'upsert')
    record(local, uri)


if __name__ == '__main__':
    which = sys.argv[1:] or ['delta', 'hudi']
    spark = session()
    try:
        if 'delta' in which:
            delta(spark)
        if 'hudi' in which:
            hudi(spark)
    finally:
        spark.stop()
    print('fixtures written to', ROOT)
