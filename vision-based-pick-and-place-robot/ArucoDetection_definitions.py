# -*- coding: utf-8 -*-
"""
ArucoDetection_definitions.py
==============================
HELPER FILE — handles ArUco boundary detection.
DO NOT RUN this file directly.
Place in the SAME FOLDER as Aruco_detection_V2_USB.py

What this file does:
  - Finds the 4 corner ArUco markers (IDs 1,2,3,4)
  - Draws green dots, ID numbers, gold overlay on boundary
  - Crops and perspective-corrects the workspace area
  - All functions are imported by the main file
"""

import cv2
import numpy as np


# ── Marker coordinate extraction ──────────────────────────────────

def getMarkerCoordinates(markers, ids, point=0):
    """
    Get a specific corner point from each detected marker.
    point: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
    """
    marker_array = []
    if markers is None or len(markers) == 0:
        return marker_array, ids
    for marker in markers:
        marker_array.append([
            int(marker[0][point][0]),
            int(marker[0][point][1])
        ])
    return marker_array, ids


def getMarkerCenter_foam(marker):
    """
    Get center point of a marker by averaging 4 corners.
    Returns [[cx, cy]] or [[0,0]] if no marker.
    """
    if marker is None or len(marker) == 0:
        return [[0, 0]]
    left_top,  _ = getMarkerCoordinates(marker, 1, point=0)
    right_top, _ = getMarkerCoordinates(marker, 1, point=1)
    left_bot,  _ = getMarkerCoordinates(marker, 1, point=2)
    right_bot, _ = getMarkerCoordinates(marker, 1, point=3)
    if bool(left_top):
        cx = (left_top[0][0]+right_top[0][0]+
              left_bot[0][0]+right_bot[0][0]) * 0.25
        cy = (left_top[0][1]+right_top[0][1]+
              left_bot[0][1]+right_bot[0][1]) * 0.25
        return [[int(cx), int(cy)]]
    return [[0, 0]]


# ── Drawing helpers ────────────────────────────────────────────────

def draw_corners(img, corners):
    """Green filled circle at each corner."""
    for corner in corners:
        cv2.circle(img, (corner[0], corner[1]),
                   10, (0, 255, 0), thickness=-1)


def draw_numbers(img, corners, ids):
    """Draw marker ID number next to each corner."""
    if ids is None or len(corners) == 0:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, corner in enumerate(corners):
        cv2.putText(img, str(ids[i]),
                    (corner[0]+10, corner[1]+10),
                    font, 2, (0, 0, 0), 4)


def show_spec(img, corners):
    """Show marker count at top-left of image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,
                str(len(corners)) + " markers found.",
                (15, 30), font, 0.8, (0, 0, 250), 2)


def draw_field(img, corners, ids):
    """
    Draw semi-transparent gold polygon between the 4 boundary markers.
    Requires markers with IDs 1, 2, 3, 4.
    Returns (annotated_image, squareFound_bool)
    """
    if ids is not None and len(corners) == 4:
        try:
            markers_sorted = [0, 0, 0, 0]
            for cid in [1, 2, 3, 4]:
                index = ids.index(cid)
                markers_sorted[cid-1] = corners[index]
            contours = np.array(markers_sorted)
            overlay  = img.copy()
            cv2.fillPoly(overlay, pts=[contours], color=(255, 215, 0))
            img_new  = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
            return img_new, True
        except (ValueError, IndexError):
            pass
    return img, False


# ── Perspective transform ──────────────────────────────────────────

def order_points(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect    = np.zeros((4, 2), dtype="float32")
    s       = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """
    Perspective warp — converts tilted workspace view
    into a flat top-down rectangle.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA   = np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))
    widthB   = np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))

    heightA   = np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
    heightB   = np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0,            0],
        [maxWidth-1,   0],
        [maxWidth-1,   maxHeight-1],
        [0,            maxHeight-1]], dtype="float32")

    M      = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
