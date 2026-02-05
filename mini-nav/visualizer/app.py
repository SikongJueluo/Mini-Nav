import datetime
from typing import Optional

import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback, dcc, html
from database import db_manager


def parse_contents(contents, filename, date):
    return html.Div(
        [
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),
            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
            html.Img(src=contents),
            html.Hr(),
            html.Div("Raw Content"),
            html.Pre(
                contents[0:200] + "...",
                style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"},
            ),
        ]
    )


class APP(Dash):
    """Singleton Dash Application"""

    _instance: Optional["APP"] = None

    def __new__(cls) -> "APP":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        super().__init__(__name__)

        df = (
            db_manager.table.search()
            .select(["id", "label", "vector"])
            .limit(1000)
            .to_polars()
        )

        columnDefs = [
            {"headerName": column.capitalize(), "field": column}
            for column in df.columns
        ]

        self.layout = dmc.MantineProvider(
            dmc.Container(
                dmc.Flex(
                    [
                        dcc.Upload(
                            id="upload-image",
                            children=html.Div(
                                ["Drag and Drop or ", html.A("Select Files")]
                            ),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "margin": "10px",
                            },
                            # Allow multiple files to be uploaded
                            multiple=True,
                        ),
                        html.Div(id="output-image-upload"),
                        dag.AgGrid(
                            rowData=df.to_dicts(),
                            columnDefs=columnDefs,
                        ),
                    ],
                    gap="md",
                    justify="center",
                    align="center",
                    direction="column",
                    wrap="wrap",
                ),
            )
        )

    @callback(
        Output("output-image-upload", "children"),
        Input("upload-image", "contents"),
        State("upload-image", "filename"),
        State("upload-image", "last_modified"),
    )
    def update_output(list_of_contents, list_of_names, list_of_dates):
        if list_of_contents is not None:
            children = [
                parse_contents(c, n, d)
                for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
            ]
            return children


app = APP()
