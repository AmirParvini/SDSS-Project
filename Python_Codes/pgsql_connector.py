import psycopg2
from psycopg2.extras import DictCursor

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
        try:
            conn = psycopg2.connect(
            self.dbname,
            self.user,
            self.password,
            self.host,
            self.port
            )
            # ایجاد یک cursor برای اجرای کوئری‌ها
            cur = conn.cursor(cursor_factory=DictCursor)
            
            cur.execute("SELECT * FROM i_d_c_s")
            idcs_rows = cur.fetchall()
            cur.execute("SELECT * FROM e_c_s")
            ecs_rows = cur.fetchall()
            cur.execute("SELECT * FROM t_m_c_s")
            tmcs_rows = cur.fetchall()
            cur.execute("SELECT * FROM damaged_areas")
            das_rows = cur.fetchall()
            cur.execute("SELECT * FROM hospitals")
            hospitals_rows = cur.fetchall()

            for ir in idcs_rows:
                for er in ecs_rows:
                    self.idc_to_ec.append({f"{ir['id']}": (ir["lng"], ir["lat"]) ,
                                           f"{er['id']}": (er["lng"], er["lat"])})
            for dr in das_rows:
                for er in ecs_rows:
                    self.da_to_ec.append({f"{dr['id']}": (dr["lng"], dr["lat"]) ,
                                           f"{er['id']}": (er["lng"], er["lat"])})
                for tr in tmcs_rows:
                    self.da_to_tmc.append({f"{dr['id']}": (dr["lng"], dr["lat"]) ,
                                           f"{tr['id']}": (tr["lng"], tr["lat"])})
                for hr in hospitals_rows:
                    self.da_to_h.append({f"{dr['id']}": (dr["lng"], dr["lat"]) ,
                                           f"{hr['id']}": (hr["lng"], hr["lat"])})
        except Exception as e:
            print(f"خطا در اتصال یا اجرای کوئری: {e}")

        finally:
            # بستن cursor و اتصال
            if cur:
                cur.close()
            if conn:
                conn.close()
                
    def data_send(self, idc_id, ec_id, path, distance, table):
        try:
            conn = psycopg2.connect(
                self.dbname,
                self.user,
                self.password,
                self.host,
                self.port
                )
                # ایجاد یک cursor برای اجرای کوئری‌ها
            cur = conn.cursor(cursor_factory=DictCursor)
            cur.execute(f"INSERT INTO {table} (idc_id, ec_id, path, distance)" + "VALUES (%s, %s, %s, %s)", (idc_id, ec_id, path, distance))
        except Exception as e:
            print(f"خطا در اتصال یا اجرای کوئری: {e}")
        finally:
            # بستن cursor و اتصال
            if cur:
                cur.close()
            if conn:
                conn.close()