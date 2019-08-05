# Adapted from https://github.com/ThomasLech/CROHME_extractor
import pickle as p
import cv2
import json
import scipy.misc
import math
import numpy as np
from skimage.draw import line
from skimage.morphology import thin
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET

# InkML namespace
doc_namespace = "{http://www.w3.org/2003/InkML}"


def parse_traces(root_element):
    # Traces are stored in <trace> nodes, and there can be multiple that
    # make up a shape. Find all traces and store by id for easy lookup
    traces = {}
    for trace_node in root_element.findall(doc_namespace + 'trace'):
        trace_id = trace_node.get('id')

        # trace segments are comma separated and can span multiple lines
        trace_segments = trace_node.text.replace('\n', '').split(',')
        processed_segments = []
        for segment in trace_segments:
            point = segment.strip().split(' ')
            x = float(point[0])
            y = float(point[1])
            processed_segments.append([x, y])

        traces[int(trace_id)] = {
            "id": trace_id,
            "segments": processed_segments
        }
    return traces


def get_traces_data(inkml_file_abs_path):
    traces_data = []

    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()

    traces_all = parse_traces(root)

    # Exit early if we were not able to load traces
    if len(traces_all) < 1:
        print("No traces!")
        return traces_data

    # Always 1st traceGroup is a redundant wrapper
    traceGroupWrapper = root.find(doc_namespace + 'traceGroup')
    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):
            label = traceGroup.find(doc_namespace + 'annotation').text

            # Traces of the current trace group
            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + 'traceView'):
                # Id reference to specific trace tag corresponding to currently considered label
                traceDataRef = int(traceView.get('traceDataRef'))

                #Each trace is represented by a list of coordinates to connect
                single_trace = traces_all[traceDataRef]['segments']
                traces_curr.append(single_trace)
            traces_data.append({'label': label, 'trace_group': traces_curr})

    else:
        # Consider Validation data that has no labels
        [traces_data.append({'trace_group': [trace['segments']]})
         for trace in traces_all]

    return traces_data

def get_groundtruth_label(inkml_file_abs_path):
    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()

    # Loop through annotations looking of the one with the ground truth label
    annotations = root.findall(doc_namespace + "annotation")
    labels = [annotation for annotation in annotations if annotation.attrib['type'] == "UI"]

    if len(labels) > 0:
        return labels[0].text
    else:
        raise ValueError("Not ground truth label found")

def convert_to_imgs(traces_data, box_size=int(100)):

    patterns_enc = []
    classes_rejected = []

    for pattern in traces_data:

        trace_group = pattern['trace_group']

        # mid coords needed to shift the pattern 
        min_x, min_y, max_x, max_y = get_min_coords(trace_group)

        # traceGroup dimensions
        trace_grp_height, trace_grp_width = max_y - min_y, max_x - min_x

        # shift pattern to its relative position
        shifted_trace_grp = shift_trace_grp(
            trace_group, min_x=min_x, min_y=min_y)

        # Interpolates a pattern so that it fits into a box with specified size'
        # method: LINEAR INTERPOLATION'
        try:
            interpolated_trace_grp = interpolate(shifted_trace_grp,
                                                 trace_grp_height=trace_grp_height, trace_grp_width=trace_grp_width, box_size=box_size - 1)
        except Exception as e:
            print(e)
            print('This data is corrupted - skipping.')
            classes_rejected.append(pattern.get('label'))

            continue

        # Get min, max coords once again in order to center scaled patter inside the box
        min_x, min_y, max_x, max_y = get_min_coords(interpolated_trace_grp)

        centered_trace_grp = center_pattern(
            interpolated_trace_grp, max_x=max_x, max_y=max_y, box_size=box_size)

        # Center scaled pattern so it fits a box with specified size
        pattern_
        n = draw_pattern(centered_trace_grp, box_size=box_size)
        # Make sure that patterns are thinned (1 pixel thick)
        pat_thinned = 1.0 - thin(1.0 - np.asarray(pattern_drawn))
        plt.imshow(pat_thinned, cmap='gray')
        plt.show()
        pattern_enc = {
            'features': pat_thinned, 
            'label': pattern.get('label')
        }

        patterns_enc.append(pattern_enc)

    return patterns_enc, classes_rejected


def get_min_coords(trace_group):
    min_x_coords = []
    min_y_coords = []
    max_x_coords = []
    max_y_coords = []

    for trace in trace_group:
        x_coords = [coord[0] for coord in trace]
        y_coords = [coord[1] for coord in trace]

        min_x_coords.append(min(x_coords))
        min_y_coords.append(min(y_coords))
        max_x_coords.append(max(x_coords))
        max_y_coords.append(max(y_coords))

    return min(min_x_coords), min(min_y_coords), max(max_x_coords), max(max_y_coords)


def shift_trace_grp(trace_group, min_x, min_y):

    shifted_trace_grp = []

    for trace in trace_group:
        shifted_trace = [[coord[0] - min_x, coord[1] - min_y]
                         for coord in trace]

        shifted_trace_grp.append(shifted_trace)

    return shifted_trace_grp


def interpolate(trace_group, trace_grp_height, trace_grp_width, box_size):
    interpolated_trace_grp = []

    if trace_grp_height == 0:
        trace_grp_height += 1
    if trace_grp_width == 0:
        trace_grp_width += 1

    #KEEP original size ratio
    trace_grp_ratio = (trace_grp_width) / (trace_grp_height)

    scale_factor = 1.0
    # Set rescale coefficient magnitude
    if trace_grp_ratio < 1.0:
        scale_factor = (box_size / trace_grp_height)
    else:
        scale_factor = (box_size / trace_grp_width)

    for trace in trace_group:
        # coordintes convertion to int type necessary
        interpolated_trace = [[round(coord[0] * scale_factor), round(coord[1] * scale_factor)] for coord in trace]

        interpolated_trace_grp.append(interpolated_trace)

    return interpolated_trace_grp


def get_min_coords(trace_group):
    min_x_coords = []
    min_y_coords = []
    max_x_coords = []
    max_y_coords = []

    for trace in trace_group:
        x_coords = [coord[0] for coord in trace]
        y_coords = [coord[1] for coord in trace]
        min_x_coords.append(min(x_coords))
        min_y_coords.append(min(y_coords))
        max_x_coords.append(max(x_coords))
        max_y_coords.append(max(y_coords))

    return min(min_x_coords), min(min_y_coords), max(max_x_coords), max(max_y_coords)



def center_pattern(trace_group, max_x, max_y, box_size):
    x_margin = int((box_size - max_x) / 2)
    y_margin = int((box_size - max_y) / 2)

    return shift_trace_grp(trace_group, min_x=-x_margin, min_y=-y_margin)


def draw_pattern(trace_group, box_size):

    pattern_drawn = np.ones(shape=(box_size, box_size), dtype=np.float32)
    for trace in trace_group:


        if len(trace) == 1:
            # Single point to draw
            x_coord = trace[0][0]
            y_coord = trace[0][1]
            pattern_drawn[y_coord, x_coord] = 0.0
        else:
            # Trace has multiple points
            for pt_idx in range(len(trace) - 1):                
                r0 = int(trace[pt_idx][1])
                r1 = int(trace[pt_idx + 1][1])
                c0 = int(trace[pt_idx][0])
                c1 = int(trace[pt_idx + 1][0])
                pattern_drawn[line(r0=r0,c0=c0,r1=r1,c1=c1)]

    return pattern_drawn


def convert_inkml_to_image(input_path, output_path):
    traces = get_traces_data(input_path)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.axes().spines['top'].set_visible(False)
    plt.axes().spines['right'].set_visible(False)
    plt.axes().spines['bottom'].set_visible(False)
    plt.axes().spines['left'].set_visible(False)
    for elem in traces:
        segment = elem['trace_group']
        for subls in segment:
            data = np.array(subls)
            x,y=zip(*data)
            plt.plot(x,y,linewidth=2,c='black')        
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.gcf().clear()


    
def latex2img(formula, fontsize='small', dpi=300, format_='svg'):
    """Renders LaTeX formula into image.
    """
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, formula, fontsize=fontsize, style='italic')
    buffer_ = StringIO()
    fig.savefig(buffer_, dpi=dpi, transparent=False,
                format=format_, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return buffer_.getvalue()


if __name__ == "__main__":
    src = "/Users/erikbeerepoot/Dropbox/project/machine-learning/projects/deeplatex/notebooks/../data/original_data/CROHME/CROHME2016/Task-2-Symbols/train/trainingSymbols/iso22771.inkml"
    tar = "/Users/erikbeerepoot/Dropbox/project/machine-learning/projects/deeplatex/notebooks/../data/processed_data/CROHME/CROHME2016/Task-2-Symbols/train/trainingSymbols/iso22771.png"
    convert_inkml_to_image(src, tar)
    
    
    
