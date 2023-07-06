import os
import numpy as np
import cv2

def gen_template_grid(uniform, increase):
    # === set uniform grid ===
    if uniform:
        # # field_dim_x, field_dim_y = 105.000552, 68.003928 # in meter
        field_dim_x, field_dim_y = 114.83, 74.37  # in yard
        nx, ny = (13, 7)
        if increase:
            nx, ny = (21, 7)
        x = np.linspace(0, field_dim_x, nx)
        y = np.linspace(0, field_dim_y, ny)
        xv, yv = np.meshgrid(x, y, indexing='ij')
        uniform_grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
        uniform_grid = np.concatenate((uniform_grid, np.ones(
            (uniform_grid.shape[0], 1))), axis=1)  # top2bottom, left2right
        # TODO: class label in template, each keypoints is (x, y, c), c is label that starts from 1
        for idx, pts in enumerate(uniform_grid):
            pts[2] = idx + 1  # keypoints label
        return uniform_grid

    ratio = 114.83 / 105
    three_stripes_width = 5.5 * ratio * 3
    five_stripes_width = 5.37 * ratio * 5
    start_five_right = 57.415 + 4.58 * ratio + 4.57 * ratio
    start_three_right = start_five_right + five_stripes_width
    x = np.array([0., three_stripes_width * 1 / 3, three_stripes_width * 2 / 3, three_stripes_width,
                  three_stripes_width + five_stripes_width * 1 / 5, three_stripes_width + five_stripes_width * 2 / 5,
                  three_stripes_width + five_stripes_width * 3 / 5, three_stripes_width + five_stripes_width * 4 / 5,
                  three_stripes_width + five_stripes_width,
                  three_stripes_width + five_stripes_width + 4.57 * ratio,
                  57.415,
                  57.415 + 4.58 * ratio, start_five_right,
                  start_five_right + five_stripes_width / 5, start_five_right + five_stripes_width * 2 / 5,
                  start_five_right + five_stripes_width * 3 / 5, start_five_right + five_stripes_width * 4 / 5,
                  start_five_right + five_stripes_width,
                  start_three_right + three_stripes_width / 3, start_three_right + three_stripes_width * 2 / 3, 114.83])
    y = np.array([0., 15.1365, 27.167, 47.203, 59.2335, 74.37])
    xv, yv = np.meshgrid(x, y, indexing='ij')
    uniform_grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
    uniform_grid = np.concatenate((uniform_grid, np.ones(
        (uniform_grid.shape[0], 1))), axis=1)

    circ1 = np.array([[12.02980952, 37.185, 1],
                      [2.02280952, 37.185, 1],
                      [22.03680952, 37.185, 1],
                      [6.01490476, 45.18256023, 1],
                      [6.01490476, 29.18743977, 1],
                      [18.04471429, 45.18256023, 1],
                      [18.04471429, 29.18743977, 1]])
    circ2 = circ1 + [45.38519048, 0, 0]
    circ3 = circ2 + [45.38519048, 0, 0]
    circ2[3:, :] = [[52.40622476, 45.84826841, 1],
                    [52.40622476, 28.52173159, 1],
                    [62.42377524, 45.84826841, 1],
                    [62.42377524, 28.52173159, 1]]

    to_concat = np.concatenate((circ1, circ2, circ3), axis=0)
    uniform_grid = np.concatenate((uniform_grid, to_concat), axis=0)
    uniform_grid = uniform_grid[np.lexsort((uniform_grid[:, 1], uniform_grid[:, 0]))]
    for idx, pts in enumerate(uniform_grid):
        pts[2] = idx + 1  # keypoints label
    return uniform_grid

def gen_template_dense_features():
    if os.path.exists("dense_template.npy"):
        return np.load("dense_template.npy")
    # field_dim_x, field_dim_y = 105.000552, 68.003928  # meters
    field_dim_x, field_dim_y = 114.83, 74.37  # in yard
    # nx, ny = (875, 567)
    nx, ny = (115, 75)

    x_resolution_yards = field_dim_x / (nx - 1)
    y_resolution_yards = field_dim_y / (ny - 1)

    marked_grid = np.zeros((ny, nx))

    # side lines
    marked_grid[0, :] = 1  # top
    marked_grid[-1, :] = 1  # bottom
    marked_grid[:, 0] = 1  # left
    marked_grid[:, -1] = 1  # right

    # mid line
    marked_grid[:, int((nx - 1) / 2)] = 1

    # penalty boxes
    penalty_box_y = round(15.1365 / y_resolution_yards) - 1
    penalty_box_x = round(18.046 / x_resolution_yards) - 1

    marked_grid[penalty_box_y, :penalty_box_x] = 1
    marked_grid[-penalty_box_y - 1, :penalty_box_x] = 1
    marked_grid[penalty_box_y:-penalty_box_y, penalty_box_x] = 1

    marked_grid[penalty_box_y, -penalty_box_x - 1:] = 1
    marked_grid[-penalty_box_y - 1, -penalty_box_x - 1:] = 1
    marked_grid[penalty_box_y:-penalty_box_y - 1, -penalty_box_x - 1] = 1

    # goal boxes
    goal_box_y = round(27.167 / y_resolution_yards) - 1
    goal_box_x = round(6.015 / x_resolution_yards) - 1

    marked_grid[goal_box_y, :goal_box_x] = 1
    marked_grid[-goal_box_y - 1, :goal_box_x] = 1
    marked_grid[goal_box_y:-goal_box_y, goal_box_x] = 1

    marked_grid[goal_box_y, -goal_box_x - 1:] = 1
    marked_grid[-goal_box_y - 1, -goal_box_x - 1:] = 1
    marked_grid[goal_box_y:-goal_box_y - 1, -goal_box_x - 1] = 1

    # semi-circles
    centre_y = int((ny - 1) / 2)
    centre_x = round(12.03 / x_resolution_yards) - 1
    radius = round(10.007 / x_resolution_yards)

    dy = np.sqrt(radius ** 2 - (penalty_box_x - centre_x) ** 2)
    angle = np.arctan(dy / (penalty_box_x - centre_x)) * 180 / np.pi

    marked_grid = cv2.ellipse(marked_grid, (centre_x, centre_y), (radius, radius), 0, -angle, angle, 1, 1)
    marked_grid = cv2.ellipse(marked_grid, (nx - centre_x - 1, centre_y), (radius, radius), 0, 180 - angle,
                              180 + angle, 1, 1)

    # center circle
    centre_y = int((ny - 1) / 2)
    centre_x = int((nx - 1) / 2)

    marked_grid = cv2.circle(marked_grid, (centre_x, centre_y), radius, 1, 1)

    marked_grid = cv2.distanceTransform(((marked_grid - 1) * -1).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    marked_grid /= marked_grid.max()

    np.save("dense_template.npy", marked_grid)

    return marked_grid
