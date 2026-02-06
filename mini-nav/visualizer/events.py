from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class CellClickedEvent:
    """
    - value (boolean I number | string I dict | list; optional): value of the clicked cell.
    - colId (boolean I number I string I dict | list; optional): column where the cell was clicked.
    - rowIndex (number; optional): rowIndex, typically a row number.
    - rowId (boolean I number I string I dict | list; optional): Row Id from the grid, this could be a number automatically, orset via getRowId.
    - timestamp (boolean I number I string I dict I list; optional): timestamp of last action.
    """

    value: Optional[Union[bool, int, float, str, dict, list]]
    """
    - value (boolean I number | string I dict | list; optional): value of the clicked cell.
    """

    colId: Optional[Union[bool, int, float, str, dict, list]]
    """
    - colId (boolean I number I string I dict | list; optional): column where the cell was clicked.
    """

    rowIndex: Optional[Union[int, float]]
    """
    - rowIndex (number; optional): rowIndex, typically a row number.
    """

    timestamp: Optional[Union[bool, int, float, str, dict, list]]
    """
    - timestamp (boolean I number I string I dict I list; optional): timestamp of last action.
    """
