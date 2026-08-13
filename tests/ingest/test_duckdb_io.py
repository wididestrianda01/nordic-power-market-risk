from nordic_power_risk.ingest.duckdb_io import get_connection, write_table


def test_get_connection_creates_parent_dirs(tmp_path):
    db_path = tmp_path / "nested" / "nordic_power_risk.duckdb"
    conn = get_connection(db_path)
    try:
        assert db_path.parent.exists()
    finally:
        conn.close()


def test_write_table_inserts_rows_and_returns_count(tmp_path):
    conn = get_connection(tmp_path / "nordic_power_risk.duckdb")
    try:
        rows = [{"timestamp": "2020-01-01T00:00:00", "value": 1.0}]
        count = write_table(conn, "raw_test", rows)
        assert count == 1
        assert conn.execute("SELECT count(*) FROM raw_test").fetchone() == (1,)
    finally:
        conn.close()


def test_write_table_replaces_existing_table(tmp_path):
    conn = get_connection(tmp_path / "nordic_power_risk.duckdb")
    try:
        write_table(conn, "raw_test", [{"value": 1.0}])
        write_table(conn, "raw_test", [{"value": 2.0}, {"value": 3.0}])
        assert conn.execute("SELECT count(*) FROM raw_test").fetchone() == (2,)
    finally:
        conn.close()


def test_write_table_handles_empty_rows(tmp_path):
    conn = get_connection(tmp_path / "nordic_power_risk.duckdb")
    try:
        count = write_table(conn, "raw_empty", [])
        assert count == 0
        assert conn.execute("SELECT count(*) FROM raw_empty").fetchone() == (0,)
    finally:
        conn.close()
