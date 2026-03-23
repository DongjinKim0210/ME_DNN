"""DB inspection utility: verify stored structural properties and responses."""
import numpy as np
from data.database import connect_db, list_tables, get_table_schema, preview_table


def inspect_db(db_path, selected_eq_id=1, selected_response_type='dsp'):
    """Print DB schema and preview structural/response data."""
    conn = connect_db(db_path)
    table_names = list_tables(conn)
    print(f"Tables: {table_names}\n")

    # Structure table
    if 'structure' in table_names:
        print("=== structure ===")
        print(get_table_schema(conn, 'structure'))
        cols, rows = preview_table(conn, 'structure', limit=3)
        if rows:
            r = rows[0]
            for c, v in zip(cols, r):
                val = v if not isinstance(v, np.ndarray) else v.tolist()
                print(f"  {c}: {val}")
        nDOF = rows[0][cols.index('nDOF')]
    else:
        print("No 'structure' table found.")
        conn.close()
        return

    # EQ table
    if 'eq_dt' in table_names:
        print("\n=== eq_dt ===")
        print(get_table_schema(conn, 'eq_dt'))
        cols_eq, rows_eq = preview_table(conn, 'eq_dt', limit=200)
        print(f"Total rows: {len(rows_eq)}")

    # Node response
    if 'node_resp' in table_names:
        print(f"\n=== node_resp (eq_id={selected_eq_id}, type={selected_response_type}) ===")
        print(get_table_schema(conn, 'node_resp'))
        cols, rows = preview_table(conn, 'node_resp', limit=500)
        filtered = [r for r in rows
                    if r[cols.index('eq_id')] == selected_eq_id
                    and r[cols.index('response_type')] == selected_response_type]
        filtered.sort(key=lambda r: r[cols.index('node_id')])
        print(f"Filtered rows: {len(filtered)}")
        for r in filtered:
            nid = r[cols.index('node_id')]
            val = r[cols.index('response_value')]
            print(f"  node_id={nid}, nPts={len(val)}, max={np.max(np.abs(val)):.4f}")

    conn.close()


def inspect_db_summary(db_path):
    """Print a compact summary of the DB contents."""
    conn = connect_db(db_path)
    tables = list_tables(conn)
    print(f"DB: {db_path}")
    print(f"Tables: {tables}")
    cur = conn.cursor()
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        count = cur.fetchone()[0]
        print(f"  {t}: {count} rows")
    conn.close()
