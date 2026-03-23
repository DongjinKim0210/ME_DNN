"""SQLite database construction and utilities with NumPy array BLOB storage."""
import os
import io
import re
import sqlite3
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from structure.fem_model import ReadRecord


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def call_adapter_converter():
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)


def connect_db(db_path: str) -> sqlite3.Connection:
    call_adapter_converter()
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB file not found: {db_path}")
    return sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)


def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
    return [r[0] for r in cur.fetchall()]


def get_table_schema(conn: sqlite3.Connection, table: str) -> str:
    cur = conn.cursor()
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table,))
    row = cur.fetchone()
    return row[0] if row else f"-- No schema found for {table}"


def preview_table(conn: sqlite3.Connection, table: str, limit: int = 5) -> Tuple[List[str], List[Tuple]]:
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table} LIMIT {limit};")
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    return cols, rows


def construct_eq_dt_table(DBfilename, eqfilepath, GMfactt, eq_label=None, delete_prior=False):
    call_adapter_converter()
    conn = sqlite3.connect(DBfilename, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    if delete_prior:
        cursor.execute('DROP TABLE IF EXISTS eq_dt')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS eq_dt (
            id INTEGER PRIMARY KEY,
            eq_name TEXT,
            dt REAL,
            nPts INTEGER,
            "acc_ts [m/s^2]" array,
            UNIQUE(eq_name)
        )
    ''')
    eqname = eq_label if eq_label else '_'.join(eqfilepath.split('/')[-2:])
    dt, nPts = ReadRecord(eqfilepath + '.AT2', eqfilepath + '.dat')
    with open(eqfilepath + '.AT2', 'r') as f:
        lines = f.readlines()
    acc_ts = np.array([float(v) for line in lines[4:] for v in line.split()]) * GMfactt
    cursor.execute('''
        INSERT OR IGNORE INTO eq_dt (eq_name, dt, nPts, "acc_ts [m/s^2]")
        VALUES (?,?,?,?)
    ''', (eqname, dt, nPts, acc_ts))
    conn.commit()
    conn.close()


def construct_structure_table(DBfilename, structure_Prop, Tnlist, ModeShapes, delete_prior=False):
    call_adapter_converter()
    conn = sqlite3.connect(DBfilename, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    if delete_prior:
        cursor.execute('DROP TABLE IF EXISTS structure')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS structure (
            id INTEGER PRIMARY KEY,
            name TEXT,
            nDOF INTEGER,
            nodal_mass_kg ARRAY,
            Ecolumn_MPa REAL,
            yieldstress_MPa REAL,
            strainhardening REAL,
            Tnlist_s ARRAY,
            ModeShapes ARRAY,
            UNIQUE(name)
        )
    ''')
    cursor.execute('''
        INSERT OR IGNORE INTO structure (name, nDOF, nodal_mass_kg, Ecolumn_MPa,
                                         yieldstress_MPa, strainhardening, Tnlist_s, ModeShapes)
        VALUES (?,?,?,?,?,?,?,?)
    ''', (structure_Prop.modelname, structure_Prop.ndof,
          np.array(structure_Prop.nodal_mass), structure_Prop.Ecolumn,
          structure_Prop.fy, structure_Prop.bb, Tnlist, ModeShapes))
    conn.commit()
    conn.close()


def construct_noderesp_table(DBfilename, dirpath, filelist, GMfact, eq_name_map=None, delete_prior=False):
    call_adapter_converter()
    conn = sqlite3.connect(DBfilename, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    if delete_prior:
        cursor.execute('DROP TABLE IF EXISTS node_resp')
    cursor.execute('PRAGMA foreign_keys=ON;')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS node_resp (
            id INTEGER PRIMARY KEY,
            eq_id INTEGER,
            GMfact REAL,
            structure_id INTEGER,
            node_id INTEGER,
            response_dof INTEGER,
            response_value ARRAY,
            analysisdt REAL,
            nPts INTEGER,
            response_type TEXT,
            FOREIGN KEY(eq_id) REFERENCES eq_dt(id),
            FOREIGN KEY(structure_id) REFERENCES structure(id),
            UNIQUE(eq_id, structure_id, node_id, response_type, response_dof)
        )
    ''')
    for file_path in tqdm(filelist, desc='[NodeResp]', unit='file'):
        normpath = os.path.normpath(os.path.join(dirpath, file_path))
        with open(normpath, 'r') as f:
            lines = f.readlines()
        dt = float(lines[0].split()[0])
        tmax = float(lines[-1].split()[0])
        analysisnPts = int(tmax / dt)
        data = [[float(v) for v in line.split()] for line in lines]
        ncols = len(data[0])
        data = [row for row in data if len(row) == ncols]
        noderespdata = np.array(data)

        match_result = re.match(r'model\((.*?)\)_inp\((.*?)\)_(.*?)\.txt', file_path)
        structure_name = match_result.group(1)
        inp_name = match_result.group(2)
        acc_or_dsp = match_result.group(3)

        cursor.execute('SELECT id FROM structure WHERE name=?', (structure_name,))
        structure_id = cursor.fetchone()[0]

        # Look up eq_name using anonymous mapping if provided
        eq_lookup = eq_name_map.get(inp_name, inp_name) if eq_name_map else inp_name
        cursor.execute('SELECT id FROM eq_dt WHERE eq_name=?', (eq_lookup,))
        row = cursor.fetchone()
        if row is None:
            continue
        inp_ts_id = row[0]

        for j in range(1, noderespdata.shape[1]):
            if acc_or_dsp in ("acc", "dsp"):
                ndid = (j - 1) // 1 + 1  # 1-indexed: recorder nodes are 1..ndof
                r_dof = (j - 1) % 1 + 1
            elif acc_or_dsp == "rct":
                ndid = (j - 1) // 2 + 1
                r_dof = (j - 1) % 2 + 1
                if r_dof == 1:
                    continue
                r_dof = 1
            cursor.execute('''
                INSERT OR IGNORE INTO node_resp
                (eq_id, GMfact, structure_id, node_id, response_dof, response_value, analysisdt, nPts, response_type)
                VALUES (?,?,?,?,?,?,?,?,?)
            ''', (inp_ts_id, GMfact, structure_id, ndid, r_dof,
                  noderespdata[:, j], dt, analysisnPts, acc_or_dsp))
    conn.commit()
    conn.close()
