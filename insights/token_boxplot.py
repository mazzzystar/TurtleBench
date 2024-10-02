import time

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ERROR AVOIDANCE BEGIN
figure = "some_figure.pdf"
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.write_image(figure, format="pdf")
time.sleep(2)
# ERROR AVOIDANCE END

# Data
# fmt: off
incorrect_lengths = [1888, 2022, 1867, 2226, 1784, 2329, 2448, 1675, 2152, 2934, 2474, 1611, 2059, 2955, 2123, 1920, 2442, 2608, 2902, 1419, 3830, 1163, 2492, 1768, 2051, 2354, 2168, 2314, 1803, 1931, 1867, 1292, 1355, 2306, 2086, 1396]
correct_lengths = [2763, 3083, 1616, 1163, 2134, 1867, 1803, 1806, 1334, 1675, 2870, 2274, 2892, 331, 2612, 1867, 2150, 1227, 1803, 2123, 1355, 2418, 1970, 2251, 1099, 1355, 1818, 3211, 1932]
# fmt: on

# Create a box plot using Plotly
fig = go.Figure()

# Adding the box plot for incorrect_lengths
fig.add_trace(
    go.Box(
        y=incorrect_lengths,
        name="Wrong Judgement",
        boxmean=True,  # display mean
        marker=dict(color="red"),
        boxpoints="all",  # show all points
    )
)

# Adding the box plot for correct_lengths
fig.add_trace(
    go.Box(
        y=correct_lengths,
        name="Right Judgement",
        boxmean=True,  # display mean
        marker=dict(color="green"),
        boxpoints="all",  # show all points
    )
)

# Update layout with tighter margins and adjusted figure size
fig.update_layout(
    yaxis_title="Number of Completion Tokens",
    font=dict(size=18),
    width=600,
    height=500,
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
    shapes=[
        dict(
            type="rect",
            x0=-0.5,
            x1=1.5*0.95,
            y0=min(min(incorrect_lengths), min(correct_lengths))*0.85,
            y1=max(max(incorrect_lengths), max(correct_lengths))*1.1,
            line=dict(color="black", width=2),
            fillcolor="rgba(0,0,0,0)",
        )
    ],
)

# Save the figure as a PDF
pio.write_image(fig, "box_plot.pdf", format="pdf")
