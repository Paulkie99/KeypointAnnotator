# image_viewer.py
import copy
import glob
import io
import os
from collections import deque
from math import sqrt

import skimage.segmentation as ss
import PySimpleGUI as sg
import numpy as np
from PIL import Image, ImageTk
import sys

from deformations import mls_affine_deformation, mls_similarity_deformation, mls_rigid_deformation
from gmc import GMC

sys.path.append('..')

import utils
import cv2

template_KPs = utils.gen_template_grid(False, False)
# template_KPs[:, :2] /= 1.09361
template_dense = 1 - utils.gen_template_dense_features()
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
GRAPH_WIDTH = 1280
GRAPH_HEIGHT = 720
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
x_neg = None
y_neg = None
KP_RADIUS = 5


def parse_folder(path):
    images = sorted(glob.glob(f'{path}/*.jpg'), key=lambda x:
    int(x.split('\\')[-1].split('.')[0].split('_')[1]) if 'IMG' in x else int(x.split('\\')[-1].split('.')[0])
                    )
    return images


def array_to_data(array):
    im = Image.fromarray(array)
    with io.BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data


def euclidean_distance(pt0, pt1):
    return sqrt((pt0[0] - pt1[0]) ** 2 + (pt0[1] - pt1[1]) ** 2)


def find_selected_KP(circle_info, x, y):
    for key in circle_info.keys():
        x_circ, y_circ = circle_info[key]["coords"]
        if abs(x - x_circ) <= KP_RADIUS and abs(y - y_circ) <= KP_RADIUS:
            return True, key
    return False, None


def load_image(path, window, gt_homo):
    global x_neg, y_neg
    try:
        window["-IMAGE-"].erase()
        image = Image.open(path)
        image_array = np.array(image, dtype=np.uint8)
        image_array = cv2.resize(image_array, (IMAGE_WIDTH, IMAGE_HEIGHT))

        S = np.eye(3)
        S[0, 0] = IMAGE_WIDTH / 1280
        S[1, 1] = IMAGE_HEIGHT / 720
        inv_homo = S @ np.linalg.inv(gt_homo)
        warped_dense = cv2.warpPerspective(template_dense, inv_homo,
                                           (IMAGE_WIDTH, IMAGE_HEIGHT),
                                           cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0)) * 255
        warped_dense[warped_dense < 240] = 0
        warped_dense = cv2.cvtColor(warped_dense, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        warped_dense[:, :, [0, 2]] = 0

        dense_alpha = 0.1
        # to_save = cv2.addWeighted(image_array, 1 - dense_alpha, warped_dense, dense_alpha, 0)
        to_save = image_array

        data = array_to_data(to_save)
        # drawing_offset_x = int((GRAPH_WIDTH - IMAGE_WIDTH) / 2)
        # drawing_offset_y = int((GRAPH_HEIGHT - IMAGE_HEIGHT) / 2)
        img_id = window["-IMAGE-"].draw_image(data=data, location=(0, 0))

        transformed_KPs = cv2.perspectiveTransform(template_KPs[:, None, :2], inv_homo).squeeze()
        circ_info = {}
        for idx, kp in enumerate(transformed_KPs):
            circ_info[idx] = {"coords": (kp[0], kp[1])}
        horizontal_lines, vertical_lines, circles = redraw_KPs_and_lines(circ_info, window, None)

        return horizontal_lines, vertical_lines, circles, img_id

    except Exception as e:
        print(e)
        raise e


def deform_KPs(circ_info, horz_lines, vert_lines, circ_id, window, new_coords, control_ps, method):
    # Find altered lines
    horz_line_id = circ_info[circ_id]["lines"][0]
    vert_line_id = circ_info[circ_id]["lines"][1]

    # Perform transformation
    orig_p = []
    q = []
    for c_id in set(horz_lines[horz_line_id]["coords"][4:] + vert_lines[vert_line_id]["coords"][4:]).union(control_ps):
        # if not circ_info[c_id]["draggable"]:
        #     continue
        if c_id == circ_id:
            q.append([new_coords[0], new_coords[1]])
        else:
            q.append([circ_info[c_id]["coords"][0], circ_info[c_id]["coords"][1]])
        orig_p.append([circ_info[c_id]["coords"][0], circ_info[c_id]["coords"][1]])
    orig_p = np.array(orig_p).reshape((-1, 2))
    q = np.array(q).reshape((-1, 2))

    vs = np.array([[circ_info[key]["coords"][0], circ_info[key]["coords"][1]] for key in circ_info.keys()]).reshape(
        (-1, 2))

    if method == 'affine':
        transformed_pts = mls_affine_deformation(vs, orig_p, q, 1.0)
    elif method == 'similarity':
        transformed_pts = mls_similarity_deformation(vs, orig_p, q, 1.0)
    elif method == 'rigid':
        transformed_pts = mls_rigid_deformation(vs, orig_p, q, 1.0)

    # Delete all lines and KPs
    for c_id in circ_info.keys():
        window["-IMAGE-"].delete_figure(circ_info[c_id]["id"])
    for d in [horz_lines, vert_lines]:
        for l_id in d.keys():
            window["-IMAGE-"].delete_figure(d[l_id]["id"])

    # Set new KP coords and draw
    for idx, c_id in enumerate(circ_info.keys()):
        circ_info[c_id]["coords"] = (transformed_pts[idx, 0], transformed_pts[idx, 1])
        color = 'white' if c_id in control_ps else 'yellow' if circ_info[c_id]['draggable'] else 'orange'
        circ_info[c_id]["id"] = window["-IMAGE-"].draw_circle(circ_info[c_id]["coords"], radius=KP_RADIUS,
                                                              line_color=color)

    # Set new line start and end points and draw
    for d, color in [(horz_lines, 'red'), (vert_lines, 'blue')]:
        for k in d.keys():
            start_circ_id = d[k]["coords"][4]
            end_circ_id = d[k]["coords"][5]
            d[k]["coords"][0] = circ_info[start_circ_id]["coords"]
            d[k]["coords"][1] = circ_info[end_circ_id]["coords"]

            x_s, y_s = d[k]["coords"][0][0], d[k]["coords"][0][1]
            x_e, y_e = d[k]["coords"][1][0], d[k]["coords"][1][1]
            d[k]["id"] = window["-IMAGE-"].draw_line((x_s, y_s), (x_e, y_e), color=color)


def compute_homo(circle_info):
    src_pts = []
    dst_pts = []
    for key in circle_info.keys():
        src_pts.append([circle_info[key]["coords"][0], circle_info[key]["coords"][1]])
        dst_pts.append(template_KPs[key, :2])

    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)

    pred_homo, _ = cv2.findHomography(
        src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2), cv2.RANSAC, 10)

    return pred_homo


def main():
    global WINDOW_WIDTH, WINDOW_HEIGHT, GRAPH_WIDTH, GRAPH_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT, x_neg, y_neg
    width_scalar = SCREEN_WIDTH / WINDOW_WIDTH
    height_scalar = SCREEN_HEIGHT / WINDOW_HEIGHT

    WINDOW_WIDTH = int(WINDOW_WIDTH * width_scalar)
    GRAPH_WIDTH = int(GRAPH_WIDTH * width_scalar)
    IMAGE_WIDTH = int(IMAGE_WIDTH * width_scalar)

    WINDOW_HEIGHT = int(WINDOW_HEIGHT * height_scalar)
    GRAPH_HEIGHT = int(GRAPH_HEIGHT * height_scalar)
    IMAGE_HEIGHT = int(IMAGE_HEIGHT * height_scalar)

    padding_sides = int((WINDOW_WIDTH - GRAPH_WIDTH) / 2)
    padding_top_bot = int((WINDOW_HEIGHT - GRAPH_HEIGHT) / 2)

    y_neg = round((GRAPH_HEIGHT - IMAGE_HEIGHT) / 2)
    x_neg = round((GRAPH_WIDTH - IMAGE_WIDTH) / 2)

    layout = [
        [
            sg.Text("Folder"),
            sg.Input("Select folder", size=(150, 1), enable_events=True, key="-FOLDER-", disabled=True),
            sg.FolderBrowse(),
        ],
        [
            sg.Text("File"),
            sg.Text("", size=(150, 1), key="-FILE-"),
            sg.Text("", size=(17, 1), key="-SAVED-", justification='center'),
            sg.Button("Prev"),
            sg.Button("Next"),
            sg.Input("Im #", size=(4, 1), key="-IMNUM-", justification="left")
        ],
        [
            [
                sg.Button("SAVE", pad=((int(650 * width_scalar), 0), (0, 0))),
                sg.Button("Predict", tooltip="Use estimated affine transformation, from previous frame to current, to "
                                             "predict KPs. Works best when custom positions were loaded in previous "
                                             "frame.")
            ],
            sg.Button("Load Original Homography", pad=((padding_sides, 0), (0, 0))),
            sg.Button("Load Custom Homography"),
            sg.Button("Load Custom Positions"),
            sg.Button("Redraw With Homography"),
        ],
        [
            sg.Text("Deformation method:", pad=((padding_sides, 0), (0, 0))),
            sg.Text("affine", key="-METHOD-"),
            sg.Text("Selected KP ID:"),
            sg.Text("", key="-KPID-"),
            sg.Text("New KP ID:"),
            sg.Input("KID", size=(3, 1), key="-NEWKPID-", justification="left")
        ],
        [
            sg.Graph((GRAPH_WIDTH, GRAPH_HEIGHT), (-x_neg, GRAPH_HEIGHT - y_neg),
                     (GRAPH_WIDTH - x_neg, -y_neg), key="-IMAGE-", enable_events=True,
                     drag_submits=True, background_color='black', motion_events=True,
                     pad=(
                         (padding_sides, 0), (0, 0)
                     ))
        ],
        [
            sg.Multiline("Keybinds:\n"
                         "P: Fix keypoint\n"
                         "V: Make keypoint draggable\n"
                         "Q: Remove keypoint\n"
                         "A: Switch to affine deformation method\n"
                         "S: Switch to similarity deformation method\n"
                         "R: Switch to rigid deformation method\n"
                         "Ctrl + Z: Undo\n"
                         "C: Create new KP (specify ID in KID box)\n"
                         "L: Lock all draggable KPs except one mouse is hovered over\n"
                         "U: Unlock all previously fixed keypoints",
                         pad=((padding_sides, 0), (0, 0)),
                         disabled=True,
                         size=(55, 11)),
        ],
    ]

    window = sg.Window("Image Viewer", layout, size=(WINDOW_WIDTH, WINDOW_HEIGHT), return_keyboard_events=True,
                       location=(0, 0))

    images = []
    location = 0

    circle_info = None
    prev_circ_info = deque(maxlen=5)
    prediction_from = None
    horz_line_info = None
    vert_line_info = None
    selected = None
    mouse_pos = None
    control_points = set()
    method = 'affine'

    affine_matx = None
    img_id = None
    gmc = GMC()

    create_KP_ID = None

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "-FOLDER-":
            images = parse_folder(values["-FOLDER-"])
            if images:
                custom_homo_path = os.path.dirname(images[0].replace('/', '\\')).replace("Dataset", "custom_labels")
                location = max(min(len(glob.glob(custom_homo_path + '\\*.png')), len(images) - 1), 0)
                prediction_from = None
                prev_circ_info.clear()
                circle_info, control_points, horz_line_info, vert_line_info, img_id = load_path(images, location,
                                                                                                window)

                img = cv2.cvtColor(cv2.cvtColor(np.array(ImageTk.getimage(window['-IMAGE-'].Images[img_id])),
                                                cv2.COLOR_RGBA2RGB), cv2.COLOR_RGB2BGR)
                affine_matx = gmc.apply(img)

                window.bind("<Control-Z>", "Control + Z")
                window.bind("<Control-z>", "Control + z")
                window["-IMNUM-"].bind("<Return>", "_Enter")
                window["-NEWKPID-"].bind("<Return>", "_Enter")

        elif event == "Next" and images:
            if location == len(images) - 1:
                location = 0
            else:
                location += 1
            prediction_from = copy.deepcopy(circle_info)
            prev_circ_info.clear()
            circle_info, control_points, horz_line_info, vert_line_info, img_id = load_path(images, location, window)

            img = cv2.cvtColor(cv2.cvtColor(np.array(ImageTk.getimage(window['-IMAGE-'].Images[img_id])),
                                            cv2.COLOR_RGBA2RGB), cv2.COLOR_RGB2BGR)
            affine_matx = gmc.apply(img)

        elif event == "Prev" and images:
            if location == 0:
                location = len(images) - 1
            else:
                location -= 1
            prediction_from = None
            prev_circ_info.clear()
            circle_info, control_points, horz_line_info, vert_line_info, img_id = load_path(images, location, window)

            img = cv2.cvtColor(cv2.cvtColor(np.array(ImageTk.getimage(window['-IMAGE-'].Images[img_id])),
                                            cv2.COLOR_RGBA2RGB), cv2.COLOR_RGB2BGR)
            affine_matx = gmc.apply(img)

        elif event == "-IMNUM-" + "_Enter":
            try:
                if values["-IMNUM-"] != '':
                    found = False
                    for idx, im_name in enumerate(images):
                        if int(im_name.split('.')[0].split('\\')[-1]) == int(values["-IMNUM-"]):
                            found = True
                            location = idx
                            break
                    if not found:
                        location = min(max(int(values["-IMNUM-"]), 0), len(images) - 1)
                    prediction_from = None
                    prev_circ_info.clear()
                    circle_info, control_points, horz_line_info, vert_line_info, img_id = load_path(images, location,
                                                                                                    window)
            except:
                window['-IMNUM-'].update(value='')

            window.force_focus()

        elif event == "-NEWKPID-" + "_Enter":
            try:
                if values["-NEWKPID-"] != '':
                    create_KP_ID = int(values["-NEWKPID-"]) - 1
                    if not (0 <= create_KP_ID <= template_KPs.shape[0]):
                        raise "Invalid KP ID"
            except:
                window['-NEWKPID-'].update(value='')
                create_KP_ID = None

            window.force_focus()

        elif "-IMAGE-" in event:
            x, y = values["-IMAGE-"]
            mouse_pos = x, y
            if event == '-IMAGE-':  # click
                if circle_info is not None:
                    isFound, key = find_selected_KP(circle_info, x, y)
                    if isFound:
                        window['-KPID-'].update(value=str(key + 1))
                        if circle_info[key]["draggable"]:
                            selected = key

            elif event == "-IMAGE-+UP":
                if selected is not None:
                    prev_circ_info.append(copy.deepcopy(circle_info))
                    deform_KPs(circle_info, horz_line_info, vert_line_info, selected, window, (x, y),
                               control_points, method)
                    selected = None

        # KEYBOARD EVENTS
        elif event == 'p' or event == 'P':
            if mouse_pos is not None and circle_info is not None:
                x, y = mouse_pos
                isFound, key = find_selected_KP(circle_info, x, y)
                if isFound:
                    control_points.add(key)
                    circle_info[key]["draggable"] = False
                    window["-IMAGE-"].delete_figure(circle_info[key]["id"])
                    circle_info[key]["id"] = window["-IMAGE-"].draw_circle(circle_info[key]["coords"],
                                                                           radius=KP_RADIUS,
                                                                           line_color='white',
                                                                           line_width=3)
        elif event == 'v' or event == 'V':
            if mouse_pos is not None and circle_info is not None:
                x, y = mouse_pos
                isFound, key = find_selected_KP(circle_info, x, y)
                if isFound:
                    if key in control_points:
                        control_points.remove(key)
                    circle_info[key]["draggable"] = True
                    window["-IMAGE-"].delete_figure(circle_info[key]["id"])
                    circle_info[key]["id"] = window["-IMAGE-"].draw_circle(circle_info[key]["coords"],
                                                                           radius=KP_RADIUS,
                                                                           line_color='yellow',
                                                                           line_width=1)
        elif event == 'a' or event == 'A':
            method = 'affine'
            window['-METHOD-'].update(value=method)
        elif event == 's' or event == 'S':
            method = 'similarity'
            window['-METHOD-'].update(value=method)
        elif event == 'r' or event == 'R':
            method = 'rigid'
            window['-METHOD-'].update(value=method)
        elif event == 'q' or event == 'Q':
            isFound, key = find_selected_KP(circle_info, mouse_pos[0], mouse_pos[1])
            if isFound:
                prev_circ_info.append(copy.deepcopy(circle_info))
                clear_KPs_and_lines(circle_info, horz_line_info, vert_line_info, window)
                circle_info.pop(key)
                horz_line_info, vert_line_info, circle_info = redraw_KPs_and_lines(circle_info, window,
                                                                                   control_ps=control_points)
        elif event == 'c' or event == 'C':
            if create_KP_ID is not None:
                x, y = mouse_pos[0], mouse_pos[1]
                prev_circ_info.append(copy.deepcopy(circle_info))
                clear_KPs_and_lines(circle_info, horz_line_info, vert_line_info, window)
                circle_info[create_KP_ID] = {"coords": (x, y)}
                horz_line_info, vert_line_info, circle_info = redraw_KPs_and_lines(circle_info, window,
                                                                                   control_ps=control_points)
        elif event == 'l' or event == 'L':  # lock all except selected KP
            if mouse_pos is not None and circle_info is not None:
                x, y = mouse_pos
                isFound, key = find_selected_KP(circle_info, x, y)
                if isFound:
                    for id_ in circle_info.keys():
                        if id_ != key and circle_info[id_]["draggable"]:
                            control_points.add(id_)
                            circle_info[id_]["draggable"] = False
                            window["-IMAGE-"].delete_figure(circle_info[id_]["id"])
                            circle_info[id_]["id"] = window["-IMAGE-"].draw_circle(circle_info[id_]["coords"],
                                                                                   radius=KP_RADIUS,
                                                                                   line_color='white',
                                                                                   line_width=3)
        elif event == 'u' or event == 'U':
            if len(control_points):
                for id_ in control_points:
                    circle_info[id_]["draggable"] = True
                    window["-IMAGE-"].delete_figure(circle_info[id_]["id"])
                    circle_info[id_]["id"] = window["-IMAGE-"].draw_circle(circle_info[id_]["coords"],
                                                                           radius=KP_RADIUS,
                                                                           line_color='yellow',
                                                                           line_width=1)

                control_points = set()

        elif event == "SAVE":
            new_homo = compute_homo(circle_info)

            path = images[location].replace('/', '\\').replace("Dataset", "custom_labels")
            gt_homo_path = path.replace('jpg', 'npy')
            os.makedirs(os.path.dirname(gt_homo_path), exist_ok=True)

            np.save(gt_homo_path, new_homo)

            kp_path = gt_homo_path.replace('npy', 'npz')
            np.savez(kp_path, circle_info, allow_pickle=True)

            # TODO save KP image
            KP_image = np.zeros((180, 320))
            for key in circle_info.keys():
                if 0 <= circle_info[key]["coords"][0] < IMAGE_WIDTH and 0 <= circle_info[key]["coords"][
                    1] < IMAGE_HEIGHT:
                    KP_image[int(circle_info[key]["coords"][1] / 4), int(circle_info[key]["coords"][0] / 4)] = key + 1
            KP_image = ss.expand_labels(KP_image, distance=KP_RADIUS)
            cv2.imwrite(path, KP_image)

            window['-SAVED-'].update(value='')
            sg.cprint("Custom saved", window=window, key="-SAVED-", c="green on white", justification='center')

        elif event == "Redraw With Homography":
            path = images[location].replace('/', '\\')
            new_homo = compute_homo(circle_info)
            prev_circ_info.append(copy.deepcopy(circle_info))
            horz_line_info, vert_line_info, circle_info, img_id = load_image(path, window, new_homo)
            control_points = set()
            window['-FILE-'].update(value=images[location])

        elif event == "Load Custom Homography":
            path = images[location].replace('/', '\\')
            gt_homo_path = path.replace("Dataset", "custom_labels").replace('jpg', 'npy')
            if os.path.exists(gt_homo_path):
                gt_homo = np.load(gt_homo_path)
                prev_circ_info.append(copy.deepcopy(circle_info))
                horz_line_info, vert_line_info, circle_info, img_id = load_image(path, window, gt_homo)
                control_points = set()
                window['-FILE-'].update(value=images[location])

        elif event == "Load Original Homography":
            path = images[location].replace('/', '\\')
            gt_homo_path = path.replace("Dataset", "Annotations").replace('jpg', 'npy')
            if os.path.exists(gt_homo_path):
                gt_homo = np.load(gt_homo_path)
                prev_circ_info.append(copy.deepcopy(circle_info))
                horz_line_info, vert_line_info, circle_info, img_id = load_image(path, window, gt_homo)
                control_points = set()
                window['-FILE-'].update(value=images[location])

        elif event == "Load Custom Positions":
            path = images[location].replace('/', '\\')
            gt_homo_path = path.replace("Dataset", "custom_labels").replace('jpg', 'npz')
            if os.path.exists(gt_homo_path):
                clear_KPs_and_lines(circle_info, horz_line_info, vert_line_info, window)

                prev_circ_info.append(copy.deepcopy(circle_info))
                circle_info = np.load(gt_homo_path, allow_pickle=True)['arr_0'][()]

                horz_line_info, vert_line_info, circle_info = redraw_KPs_and_lines(circle_info, window, None)

        elif event == 'Predict':
            if prediction_from is not None:
                prev_circ_info.append(copy.deepcopy(circle_info))

                for id in circle_info.keys():
                    if id not in prediction_from.keys():
                        circle_info[id]["mismatch"] = True

                for id, kp in prediction_from.items():
                    old_coord = np.array([[kp["coords"][0]],
                                          [kp["coords"][1]],
                                          [1]])  # 3 x 1
                    new_coord = affine_matx @ old_coord  # 2 x 1
                    if id not in circle_info.keys():
                        circle_info[id] = {
                            "coords": tuple(new_coord.flatten()),
                            "mismatch": True,
                            "id": None
                        }
                    else:
                        circle_info[id]["coords"] = tuple(new_coord.flatten())

                clear_KPs_and_lines(circle_info, horz_line_info, vert_line_info, window)

                horz_line_info, vert_line_info, circle_info = redraw_KPs_and_lines(circle_info, window, None)

                prediction_from = None

        elif event == 'Control + Z' or event == 'Control + z':
            if len(prev_circ_info):
                clear_KPs_and_lines(circle_info, horz_line_info, vert_line_info, window)
                circle_info = prev_circ_info.pop()
                horz_line_info, vert_line_info, circle_info = redraw_KPs_and_lines(circle_info, window, None)

    window.close()


def redraw_KPs_and_lines(circle_info, window, control_ps=None):
    global x_neg, y_neg

    # Check line start and end points
    horizontal_lines = {}
    vertical_lines = {}
    circles = {}

    any_mismatch = False

    for idx in circle_info.keys():  # Get lines
        x, y = circle_info[idx]["coords"][0], circle_info[idx]["coords"][1]

        if (not -x_neg <= x < int(GRAPH_WIDTH - x_neg)) or (not -y_neg <= y < int(GRAPH_HEIGHT - y_neg)):
            continue

        if "mismatch" in circle_info[idx].keys() and circle_info[idx]["mismatch"]:
            any_mismatch = True

        horizontal_id = round(template_KPs[idx, 1], 1)  # y coord specifying horizontal line
        vertical_id = round(template_KPs[idx, 0], 1)  # x coord specifying vertical line

        for d, d_key, d_check in [(horizontal_lines, horizontal_id, vertical_id),
                                  (vertical_lines, vertical_id, horizontal_id)]:
            if d_key not in d.keys():
                d[d_key] = {
                    "coords": [(x, y), (x, y), d_check, d_check, idx, idx],
                    # START END START_X END_X, START_ID, END_ID
                    "KPs": {idx}
                }
            elif d_check < d[d_key]["coords"][2]:
                d[d_key]["coords"][0] = (x, y)
                d[d_key]["coords"][2] = d_check
                d[d_key]["coords"][4] = idx
            elif d_check > d[d_key]["coords"][3]:
                d[d_key]["coords"][1] = (x, y)
                d[d_key]["coords"][3] = d_check
                d[d_key]["coords"][5] = idx

            d[d_key]["KPs"].add(idx)
        circles[idx] = {
            "coords": (x, y),
            "lines": (horizontal_id, vertical_id),
            "mismatch": True if ("mismatch" in circle_info[idx].keys() and circle_info[idx]["mismatch"]) else False
        }

    circle_info = circles

    # Set new line start and end points and draw
    for d, color in [(horizontal_lines, 'red'), (vertical_lines, 'blue')]:
        for k in d.keys():
            start_circ_id = d[k]["coords"][4]
            end_circ_id = d[k]["coords"][5]
            d[k]["coords"][0] = circle_info[start_circ_id]["coords"]
            d[k]["coords"][1] = circle_info[end_circ_id]["coords"]

            x_s, y_s = d[k]["coords"][0][0], d[k]["coords"][0][1]
            x_e, y_e = d[k]["coords"][1][0], d[k]["coords"][1][1]
            d[k]["id"] = window["-IMAGE-"].draw_line((x_s, y_s), (x_e, y_e), color=color)

    # Draw new KPs
    for idx in circle_info.keys():
        horizontal_id = round(template_KPs[idx, 1], 1)  # y coord specifying horizontal line
        vertical_id = round(template_KPs[idx, 0], 1)  # x coord specifying vertical line

        x, y = circle_info[idx]["coords"][0], circle_info[idx]["coords"][1]

        draggable = (x, y) in vertical_lines[vertical_id]["coords"][:2] or (x, y) in \
                    horizontal_lines[horizontal_id]["coords"][:2]
        if horizontal_lines[circles[idx]["lines"][0]]["coords"][0] == \
                horizontal_lines[circles[idx]["lines"][0]]["coords"][1]:
            draggable = False
        if vertical_lines[circles[idx]["lines"][1]]["coords"][0] == \
                vertical_lines[circles[idx]["lines"][1]]["coords"][1]:
            draggable = False

        if control_ps is not None and idx in control_ps:
            circles[idx]["draggable"] = False
            color = 'white'
        else:
            circles[idx]["draggable"] = draggable
            color = 'yellow' if circle_info[idx]['draggable'] else 'orange'

        width = 1
        if circle_info[idx]["mismatch"]:
            color = 'red'
            width = 3
        circle_info[idx]["id"] = window["-IMAGE-"].draw_circle(circle_info[idx]["coords"],
                                                               radius=KP_RADIUS,
                                                               line_color=color, line_width=width)

    if any_mismatch:
        sg.popup("Keypoint mismatch between current and previous frames, check red keypoint(s).")

    return horizontal_lines, vertical_lines, circle_info


def clear_KPs_and_lines(circle_info, horz_line_info, vert_line_info, window):
    # Delete all lines and KPs
    for c_id in circle_info.keys():
        if circle_info[c_id]["id"] is not None:
            window["-IMAGE-"].delete_figure(circle_info[c_id]["id"])
    for d in [horz_line_info, vert_line_info]:
        for l_id in d.keys():
            window["-IMAGE-"].delete_figure(d[l_id]["id"])


def load_path(images, location, window):
    path = images[location].replace('/', '\\')
    if 'soccer_worldcup_2014' in path:  # WC dataset
        gt_homo = np.loadtxt(path.replace('.jpg', '.homographyMatrix'))
    else:
        gt_homo_path = path.replace("Dataset", "Annotations")
        if 'IMG' in gt_homo_path:
            gt_homo = np.load(gt_homo_path.replace('.jpg', '_homography.npy'))
            horz_line_info, vert_line_info, circle_info, img_id = load_image(path, window, gt_homo)
        else:
            gt_homo = np.load(gt_homo_path.replace('.jpg', '.npy'))
            horz_line_info, vert_line_info, circle_info, img_id = load_image(path, window, gt_homo)
            clear_KPs_and_lines(circle_info, horz_line_info, vert_line_info, window)
            circle_info = np.load(gt_homo_path.replace('jpg', 'npz'), allow_pickle=True)['arr_0'][()]
            horz_line_info, vert_line_info, circle_info = redraw_KPs_and_lines(circle_info, window, None)

    control_points = set()
    window['-FILE-'].update(value=images[location])

    gt_pos_path = path.replace("Dataset", "custom_labels").replace('jpg', 'npz')

    window['-SAVED-'].update(value='')
    if os.path.exists(gt_pos_path):
        sg.cprint("Custom saved", window=window, key="-SAVED-", c="green on white", justification='center')
    else:
        sg.cprint("Custom not saved", window=window, key="-SAVED-", c="red on black", justification='center')

    return circle_info, control_points, horz_line_info, vert_line_info, img_id


if __name__ == "__main__":
    main()
