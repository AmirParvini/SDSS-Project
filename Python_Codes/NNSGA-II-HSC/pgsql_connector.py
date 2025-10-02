import psycopg2
from psycopg2.extras import RealDictCursor
import json

class PgsqlConnector():
    
    def __init__(self, dbname, user, password, host, port):   
        self.idc_to_ec = []
        self.da_to_ec = []
        self.da_to_tmc = []
        self.da_to_h = []
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    def data_fetch(self):  
        conn = psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
            )
        # ایجاد یک cursor برای اجرای کوئری‌ها
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("SELECT DISTINCT * FROM i_d_c_s")
        idcs_rows = cur.fetchall()
        cur.execute("SELECT DISTINCT * FROM e_c_s")
        ecs_rows = cur.fetchall()
        cur.execute("SELECT DISTINCT * FROM t_m_c_s")
        tmcs_rows = cur.fetchall()
        cur.execute("SELECT DISTINCT * FROM damaged_areas")
        das_rows = cur.fetchall()
        cur.execute("SELECT DISTINCT * FROM hospitals")
        hospitals_rows = cur.fetchall()

        for ir in idcs_rows:
            for er in ecs_rows:
                self.idc_to_ec.append({"source_id": ir['id'], "source_coord": (ir["lng"], ir["lat"]),
                                       "target_id": er['id'], "target_coord": (er["lng"], er["lat"])})
        for dr in das_rows:
            for er in ecs_rows:
                self.da_to_ec.append({"source_id": dr['id'], "source_coord": (dr["lng"], dr["lat"]),
                                       "target_id": er['id'], "target_coord": (er["lng"], er["lat"])})
            for tr in tmcs_rows:
                self.da_to_tmc.append({"source_id": dr['id'], "source_coord": (dr["lng"], dr["lat"]),
                                       "target_id": tr['id'], "target_coord": (tr["lng"], tr["lat"])})
            for hr in hospitals_rows:
                self.da_to_h.append({"source_id": dr['id'], "source_coord": (dr["lng"], dr["lat"]),
                                       "target_id": hr['id'], "target_coord": (hr["lng"], hr["lat"])})
        if cur:
            cur.close()
        if conn:
            conn.close()
            
        return self.idc_to_ec, self.da_to_ec, self.da_to_tmc, self.da_to_h
                
    def data_send(self, source_id, target_id, path, distance, table):
        conn = psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
            )
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        if table == "idc_ec_path":
            cur.execute(f"INSERT INTO {table} (idc_id, ec_id, distance, path)" + "VALUES (%s, %s, %s, %s)", (source_id, target_id, distance, path))
        elif table == "da_h_path":
            cur.execute(f"INSERT INTO {table} (da_id, h_id, distance, path)" + "VALUES (%s, %s, %s, %s)", (source_id, target_id, distance, path))
        elif table == "da_tmc_path":
            cur.execute(f"INSERT INTO {table} (da_id, tmc_id, distance, path)" + "VALUES (%s, %s, %s, %s)", (source_id, target_id, distance, path))
        
        conn.commit()
        # بستن cursor و اتصال
        if cur:
            cur.close()
        if conn:
            conn.close()