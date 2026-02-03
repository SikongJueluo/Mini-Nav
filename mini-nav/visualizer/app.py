from typing import Optional

import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash import Dash
from database import db_manager


class APP(Dash):
    """Singleton Dash Application"""

    _instance: Optional["APP"] = None

    def __new__(cls) -> "APP":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        super().__init__(__name__)

        df = db_manager.table.search().select(["id", "label", "vector"]).to_polars()

        columnDefs = [
            {"headerName": column.capitalize(), "field": column}
            for column in df.columns
        ]

        self.layout = dmc.MantineProvider(
            dag.AgGrid(
                rowData=df.to_dicts(),
                columnDefs=columnDefs,
            )
        )


app = APP()
