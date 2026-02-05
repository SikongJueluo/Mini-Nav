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
                            id="ag-grid",
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
        Output("ag-grid", "rowData"),
        Input("upload-image", "contents"),
        State("upload-image", "filename"),
        State("upload-image", "last_modified"),
    )
    def update_output(
        list_of_contents: List[str],
        list_of_names: List[str],
        list_of_dates: List[int] | List[float],
    ):
        def parse_base64_to_pil(contents: str) -> Image.Image:
            """Parse base64 string to PIL Image."""
            # Remove data URI prefix (e.g., "data:image/png;base64,")
            base64_str = contents.split(",")[1]
            img_bytes = base64.b64decode(base64_str)
            return Image.open(io.BytesIO(img_bytes))

        if list_of_contents is not None:
            # Process first uploaded image for similarity search
            filename = list_of_names[0]
            uploaddate = list_of_dates[0]
            imagecontent = list_of_contents[0]

            pil_image = parse_base64_to_pil(imagecontent)

            # Extract feature vector using DINOv2
            feature_vector = APP._feature_retrieval.extract_single_image_feature(
                pil_image
            )

            # Search for similar images in database
            results_df = (
                db_manager.table.search(feature_vector)
                .select(["id", "label", "vector"])
                .limit(10)
                .to_polars()
            )

            # Convert to AgGrid row format
            row_data = results_df.to_dicts()

            # Display uploaded images
            children = [
                html.H5(filename),
                html.H6(str(datetime.datetime.fromtimestamp(uploaddate))),
                # HTML images accept base64 encoded strings in same format
                # that is supplied by the upload
                dmc.Image(src=imagecontent),
                dmc.Text(f"{feature_vector[:5]}", size="xs"),
            ]

            return children, row_data

        # Return empty if no content
        return [], []


app = APP()
