import dash
from dash import dcc, html
import dash_daq as daq

slyder_style = {"font-size": "20px", "text-align": "center"}
slyder_div_style = {"display": "flex", "flex-direction": "column", "width": "75%", "gap": "5px"}
switch_div_style = {"display": "flex", "gap": "5px", "justify-content": "flex-start", "width": "18%"}
images_div_style = {"display": "flex", "align-items": "flex-start", "justify-content": " space-evenly", "width": "100%"}
main_div_style = {"display": "flex", "align-items": "center", "gap": "10px", "justify-content": "center", "flex-direction": "column", "height": "100%"}


def image_block(place, weight_text, default_index=0):
    return html.Div(
        [
            html.Img(id=f"{place}-image", style={"width": "100%"}),
            html.Div(
                [
                    html.Div(id=f"idx-{place}-description", children=f"Image n°", style={"font-size": "15px"}),
                    daq.NumericInput(id=f"{place}-image-index", value=default_index),
                    html.Div(id=f"{place}-weight-text", children=weight_text, style={"font-size": "15px"}),
                ],
                style={"display": "flex", "align-items": "center", "gap": "10px"},
            ),
        ],
        style={"display": "flex", "align-items": "center", "flex-direction": "column", "width": "25%", "gap": "10px"},
    )
