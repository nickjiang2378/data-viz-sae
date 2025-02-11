import numpy as np
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, HoverTool, LabelSet, CustomJS, Slider, Div
)
from bokeh.layouts import column, row
import torch
import scipy.sparse
import streamlit as st

# ---------------------------------------------------------------
# 1) Load your data and build expansions_map
# ---------------------------------------------------------------
ind_to_question = torch.load("./feature_viz/ind_to_question.pt")

rand_activations_sparse = scipy.sparse.load_npz("./feature_viz/rand_activations_sparse.npz")
feature_activations = rand_activations_sparse.toarray()

feature_labels = torch.load("./feature_viz/feature_labels.pt")

nonzero_features = (feature_activations[:, :max(feature_labels.keys())] > 0).sum(axis=0)
top_features = np.argsort(nonzero_features)[::-1]

texts = []
numbers = []
expansions_map = {}

for i in top_features[100:]:
    if i in feature_labels and nonzero_features[i] > 50:
        label = feature_labels[i]
        # If it's a brand-new label, add it
        if label not in texts:
            texts.append(label)
            numbers.append(nonzero_features[i])
            expansions_map[label] = []
            for j in np.argsort(feature_activations[:, i])[::-1][:10]:
                if feature_activations[j, i] > 0.1:
                    expansions_map[label].append(ind_to_question[j])
        # If this label already exists, update the count if bigger
        else:
            idx = texts.index(label)
            numbers[idx] = max(numbers[idx], nonzero_features[i])

# ---------------------------------------------------------------
# 2) Load precomputed 2D coords
# ---------------------------------------------------------------
coords_2d = torch.load("./feature_viz/coords_2d.pt")["coords_2d"]
x_coords = coords_2d[:, 0]
y_coords = coords_2d[:, 1]

# We assume x_coords, y_coords line up with 'texts' and 'numbers'.
# Make sure lengths match or slice them accordingly. For demonstration,
# let's just slice if there's a mismatch (this depends on how you computed coords).
min_len = min(len(texts), len(x_coords))
texts = texts[:min_len]
numbers = numbers[:min_len]
x_coords = x_coords[:min_len]
y_coords = y_coords[:min_len]

circle_sizes = [n/2 for n in numbers]
truncated_texts = [t[:20] + "..." if len(t) > 20 else t for t in texts]

# ---------------------------------------------------------------
# 3) Build expansions_list so each row has its expansions
# ---------------------------------------------------------------
expansions_list = []
for t in texts:
    expansions_list.append(expansions_map.get(t, []))

# ---------------------------------------------------------------
# 4) Full source + filtered source
#    Must include 'expansions' in data so JavaScript can access it
# ---------------------------------------------------------------
full_source = ColumnDataSource(data=dict(
    x=x_coords,
    y=y_coords,
    text=texts,
    truncated=truncated_texts,
    number=numbers,
    size=circle_sizes,
    expansions=expansions_list
))

source = ColumnDataSource(data=full_source.data.copy())

# A Div to show expansions
div = Div(
    text="<i>Click a bubble to see details...</i>",
    width=400,
    height=200,
    style={"margin-left": "20px", "font-size": "12px"}
)

# ---------------------------------------------------------------
# 5) Pure JS callback for tapping a circle
# ---------------------------------------------------------------
callback_js = CustomJS(
    args=dict(source=source, div=div),
    code="""
    const inds = source.selected.indices;
    if (inds.length === 0) {
        div.text = "<i>Click a bubble to see details...</i>";
        return;
    }
    const i = inds[0];
    const expansions = source.data['expansions'][i];  // array of strings
    const text = source.data['text'][i];

    if (!expansions || expansions.length === 0) {
        div.text = `<b>No expansions found for:</b> ${text}`;
    } else {
        let expansions_html = "";
        for (let e of expansions) {
            expansions_html += `<li>${e}</li>`;
        }
        div.text = `<div style="font-size: 12px;">${text}</div><ul style="font-size: 12px;">${expansions_html}</ul>`;
    }
"""
)

# Fire the callback when selection changes
source.selected.js_on_change("indices", callback_js)

# ---------------------------------------------------------------
# 6) Create the figure
# ---------------------------------------------------------------
p = figure(
    title="Text Bubbles by Semantic Similarity",
    width=1000,
    height=1000,
    tools="pan,wheel_zoom,box_zoom,reset,tap",
    active_scroll="wheel_zoom"
)
# Add circles
p.circle(
    x="x",
    y="y",
    size="size",
    source=source,
    fill_alpha=0.6,
    line_color="black",
    selection_color="red"
)
# Hover tool
hover = HoverTool(
    tooltips=[("Text", "@text"), ("Number", "@number")]
)
p.add_tools(hover)

# Optional labels
labels = LabelSet(
    x="x",
    y="y",
    level="glyph",
    x_offset=5,
    y_offset=5,
    text="truncated",
    source=source,
    visible=False
)
p.add_layout(labels)

# ---------------------------------------------------------------
# 7) Auto-hide labels unless zoomed in
# ---------------------------------------------------------------
threshold_x = 5
threshold_y = 5
zoom_code = """
const range_width = p.x_range.end - p.x_range.start
const range_height = p.y_range.end - p.y_range.start
if (range_width < threshold_x && range_height < threshold_y) {
    labelset.visible = true
} else {
    labelset.visible = false
}
"""
zoom_callback = CustomJS(
    args=dict(p=p, labelset=labels, threshold_x=threshold_x, threshold_y=threshold_y),
    code=zoom_code
)
p.x_range.js_on_change("start", zoom_callback)
p.x_range.js_on_change("end", zoom_callback)
p.y_range.js_on_change("start", zoom_callback)
p.y_range.js_on_change("end", zoom_callback)

# ---------------------------------------------------------------
# 8) Slider to filter points by 'number'
# ---------------------------------------------------------------
min_nonzero = 5
max_nonzero = max(numbers) if numbers else 100

slider = Slider(
    start=min_nonzero,
    end=max_nonzero,
    value=min_nonzero,
    step=1,
    title="Minimum Non-Zero Features"
)

slider_code = """
var threshold = cb_obj.value;
var full_data = full_source.data;
var s_data = source.data;

var newX = [];
var newY = [];
var newText = [];
var newTruncated = [];
var newNumber = [];
var newSize = [];
var newExpansions = [];

for (var i = 0; i < full_data['x'].length; i++) {
    if (full_data['number'][i] >= threshold) {
        newX.push(full_data['x'][i]);
        newY.push(full_data['y'][i]);
        newText.push(full_data['text'][i]);
        newTruncated.push(full_data['truncated'][i]);
        newNumber.push(full_data['number'][i]);
        newSize.push(full_data['size'][i]);
        newExpansions.push(full_data['expansions'][i]);
    }
}

s_data['x'] = newX;
s_data['y'] = newY;
s_data['text'] = newText;
s_data['truncated'] = newTruncated;
s_data['number'] = newNumber;
s_data['size'] = newSize;
s_data['expansions'] = newExpansions;

source.change.emit();
"""
slider_callback = CustomJS(
    args=dict(source=source, full_source=full_source),
    code=slider_code
)
slider.js_on_change("value", slider_callback)

# ---------------------------------------------------------------
# 9) Final layout and Streamlit
# ---------------------------------------------------------------
layout = row(column(slider, p), div)

st.title("Data Viz w/ SAEs")
st.bokeh_chart(layout, use_container_width=True)
