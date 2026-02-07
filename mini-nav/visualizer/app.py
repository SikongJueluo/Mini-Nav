import base64
import io
from typing import List, Optional

import dash_ag_grid as dag
import dash_mantine_components as dmc
import numpy as np
from dash import Dash, Input, Output, State, callback, dcc, html
from database import db_manager
from feature_retrieval import FeatureRetrieval
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from visualizer.events import CellClickedEvent


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
                                html.Div(
                                    [
                                        html.H4(
                                            "Uploaded Image",
                                            style={"textAlign": "center"},
                                        ),
                                        html.Div(
                                            id="output-image-upload",
                                            style={
                                                "minWidth": "300px",
                                                "minHeight": "300px",
                                                "display": "flex",
                                                "flexDirection": "column",
                                                "alignItems": "center",
                                                "justifyContent": "center",
                                                "border": "1px dashed #ccc",
                                                "borderRadius": "5px",
                                                "padding": "10px",
                                            },
                                        ),
                                    ],
                                    style={"flex": 1, "maxWidth": "45%"},
                                ),
                                dmc.Divider(
                                    variant="solid", orientation="vertical", size="sm"
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Selected Image",
                                            style={"textAlign": "center"},
                                        ),
                                        html.Div(
                                            id="output-image-select",
                                            style={
                                                "minWidth": "300px",
                                                "minHeight": "300px",
                                                "display": "flex",
                                                "flexDirection": "column",
                                                "alignItems": "center",
                                                "justifyContent": "center",
                                                "border": "1px dashed #ccc",
                                                "borderRadius": "5px",
                                                "padding": "10px",
                                            },
                                        ),
                                    ],
                                    style={"flex": 1, "maxWidth": "45%"},
                                ),
                            ],
                            gap="md",
                            justify="center",
                            align="stretch",
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
    image_content: Optional[str],
    filename: Optional[str],
    timestamp: Optional[int | float],
):
    def parse_base64_to_pil(contents: str) -> Image.Image:
        """Parse base64 string to PIL Image."""
        # Remove data URI prefix (e.g., "data:image/png;base64,")
        base64_str = contents.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_bytes))

    if image_content is not None and filename is not None and timestamp is not None:
        pil_image = parse_base64_to_pil(image_content)

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
            # HTML images accept base64 encoded strings in same format
            # that is supplied by the upload
            dmc.Image(src=image_content, w="100%", h="auto"),
            dmc.Text(f"{feature_vector[:5]}", size="xs"),
        ]

        return children, row_data, columnDefs
    else:
        # When contents is None
        # Exclude 'vector' and 'binary' columns as they are not JSON serializable
        df = db_manager.table.search().select(["id", "label"]).limit(100).to_polars()

        row_data = df.to_dicts()

        columnDefs = [
            {"headerName": column.capitalize(), "field": column}
            for column in df.columns
        ]

        return [], row_data, columnDefs


@callback(
    Output("output-image-select", "children"),
    Input("ag-grid", "cellClicked"),
    State("ag-grid", "rowData"),
)
def update_images_comparison(
    clicked_event: Optional[CellClickedEvent],
    row_data: Optional[List[dict]],
):
    if (
        clicked_event is None
        or clicked_event["rowIndex"] is None
        or row_data is None
        or len(row_data) == 0
    ):
        return []

    # Get the selected row's data
    row_index = int(clicked_event["rowIndex"])
    if row_index >= len(row_data):
        return []

    selected_row = row_data[row_index]
    image_id = selected_row.get("id")

    if image_id is None:
        return []

    # Query database for binary data using the id
    result = (
        db_manager.table.search()
        .where(f"id = {image_id}")
        .select(["id", "label", "vector", "binary"])
        .limit(1)
        .to_polars()
    )

    if result.height == 0:
        return []

    # Get binary data
    binary_data = result.row(0, named=True)["binary"]
    vector = result.row(0, named=True)["vector"]

    # Try to detect if binary_data is a valid image format (PNG/JPEG) or raw pixels
    # PNG files start with bytes: 89 50 4E 47 (hex) = \x89PNG
    # JPEG files start with bytes: FF D8 FF
    is_png = binary_data[:4] == b"\x89PNG"
    is_jpeg = binary_data[:3] == b"\xff\xd8\xff"

    if is_png or is_jpeg:
        # Binary data is already in a valid image format
        mime_type = "image/png" if is_png else "image/jpeg"
        base64_str = base64.b64encode(binary_data).decode("utf-8")
        image_content = f"data:{mime_type};base64,{base64_str}"
    else:
        # Legacy format: raw pixel bytes (CIFAR-10 images are 32x32 RGB)
        img_array = np.frombuffer(binary_data, dtype=np.uint8).reshape(32, 32, 3)
        pil_image = Image.fromarray(img_array)

        # Encode as PNG to get proper image format
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Convert to base64 with correct MIME type
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_content = f"data:image/png;base64,{base64_str}"

    # Display selected image
    children = [
        dmc.Image(src=image_content, w="100%", h="auto"),
        dmc.Text(f"{vector[:5]}", size="xs"),
    ]

    return children


app = APP()
