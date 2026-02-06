import base64
import datetime
import io
from typing import List, Optional

import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback, dcc, html
from database import db_manager
from feature_retrieval import FeatureRetrieval
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from .events import CellClickedEvent


class APP(Dash):
    """Singleton Dash Application"""

    _instance: Optional["APP"] = None

    # Feature retrieval singleton
    _feature_retrieval: FeatureRetrieval

    def __new__(cls) -> "APP":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        super().__init__(__name__)

        # Initialize FeatureRetrieval
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        model = AutoModel.from_pretrained("facebook/dinov2-large")
        APP._feature_retrieval = FeatureRetrieval(processor, model)

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
                            # Disallow multiple files to be uploaded
                            multiple=False,
                        ),
                        dmc.Flex(
                            [
                                html.Div(id="output-image-upload"),
                                html.Div(id="output-image-select"),
                            ],
                            gap="md",
                            justify="center",
                            align="center",
                            direction="row",
                            wrap="wrap",
                        ),
                        dag.AgGrid(id="ag-grid"),
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
    Output("ag-grid", "rowData"),
    Output("ag-grid", "columnDefs"),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
    State("upload-image", "last_modified"),
)
def update_output(
    list_of_contents: Optional[List[str]],
    list_of_names: Optional[List[str]],
    list_of_dates: Optional[List[int] | List[float]],
):
    def parse_base64_to_pil(contents: str) -> Image.Image:
        """Parse base64 string to PIL Image."""
        # Remove data URI prefix (e.g., "data:image/png;base64,")
        base64_str = contents.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_bytes))

    if (
        list_of_contents is not None
        and list_of_names is not None
        and list_of_dates is not None
    ):
        # Process first uploaded image for similarity search
        filename = list_of_names[0]
        uploaddate = list_of_dates[0]
        imagecontent = list_of_contents[0]

        pil_image = parse_base64_to_pil(imagecontent)

        # Extract feature vector using DINOv2
        feature_vector = APP._feature_retrieval.extract_single_image_feature(pil_image)

        # Search for similar images in database
        # Exclude 'vector' and 'binary' columns as they are not JSON serializable
        results_df = (
            db_manager.table.search(feature_vector)
            .select(["id", "label"])
            .limit(10)
            .to_polars()
        )

        # Convert to AgGrid row format
        row_data = results_df.to_dicts()

        columnDefs = [
            {"headerName": column.capitalize(), "field": column}
            for column in results_df.columns
        ]

        # Display uploaded images
        children = [
            html.H5(filename),
            html.H6(str(datetime.datetime.fromtimestamp(uploaddate))),
            # HTML images accept base64 encoded strings in same format
            # that is supplied by the upload
            dmc.Image(src=imagecontent),
            dmc.Text(f"{feature_vector[:5]}", size="xs"),
        ]

        return children, row_data, columnDefs
    else:
        # When contents is None
        # Exclude 'vector' and 'binary' columns as they are not JSON serializable
        df = db_manager.table.search().select(["id", "label"]).limit(1000).to_polars()

        row_data = df.to_dicts()

        columnDefs = [
            {"headerName": column.capitalize(), "field": column}
            for column in df.columns
        ]

        return [], row_data, columnDefs


@callback(
    Input("ag-grid", "cellClicked"),
    State("ag-grid", "row_data"),
    Output("output-image-select", "children"),
)
def update_images_comparison(
    clicked_event: Optional[CellClickedEvent], row_data: Optional[dict]
):
    if clicked_event is None or CellClickedEvent.rowIndex is None or row_data is None:
        return []

    return []


app = APP()
