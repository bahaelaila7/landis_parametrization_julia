import pyarrow as pa
import glob
import duckdb

# Method 1: Using memory_map (Recommended for performance)
files = sorted(glob.glob('outputs/landis_test/cohorts_year*.arrow'))[::-1]
table_created = False
with duckdb.connect('outputs/landis_test/cohorts.duckdb') as con:
    for file in files:
        with pa.memory_map(file, 'r') as source:
            table = pa.ipc.open_file(source).read_all()
            con.register('ar_table',table)
            if not table_created:
                con.sql('CREATE OR REPLACE TABLE output_community AS SELECT * FROM ar_table')
                table_created = True
            else:
                con.sql('INSERT INTO output_community FROM ar_table')
            con.unregister('ar_table')

