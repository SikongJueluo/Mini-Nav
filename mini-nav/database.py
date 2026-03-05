from typing import Optional

import lancedb
import pyarrow as pa
from configs import cfg_manager


def _build_database_schema():
    return pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("label", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 1024)),
            pa.field("binary", pa.binary()),
        ]
    )


class DatabaseManager:
    """Singleton Database manager"""

    _instance: Optional["DatabaseManager"] = None
    db: lancedb.DBConnection
    table: lancedb.Table

    def __new__(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 获取数据库位置
        config = cfg_manager.get()
        db_path = config.output.directory / "database"

        # 初始化数据库与表格
        self.db = lancedb.connect(db_path)
        if "default" not in self.db.list_tables().tables:
            self.table = self.db.create_table(
                "default", schema=_build_database_schema()
            )
        else:
            self.table = self.db.open_table("default")


db_manager = DatabaseManager()
