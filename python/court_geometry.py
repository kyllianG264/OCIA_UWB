import math

import pygame


COURT_LENGTH_CM = 2800.0
COURT_WIDTH_CM = 1500.0
CENTER_CIRCLE_RADIUS_CM = 180.0
THREE_POINT_RADIUS_CM = 675.0
FREE_THROW_RADIUS_CM = 180.0
RESTRICTED_RADIUS_CM = 125.0
HOOP_OFFSET_FROM_ENDLINE_CM = 157.5
BACKBOARD_OFFSET_FROM_ENDLINE_CM = 120.0
BACKBOARD_WIDTH_CM = 180.0
KEY_WIDTH_CM = 490.0
KEY_LENGTH_CM = 580.0
CORNER_THREE_OFFSET_CM = 90.0

COURT_WOOD = (196, 140, 86)
COURT_WOOD_ALT = (206, 150, 96)
COURT_APRON = (64, 82, 68)
COURT_LINES = (248, 245, 236)
COURT_PAINT = (214, 121, 72)
COURT_CENTER_LOGO = (160, 88, 54)


def court_bounds(center_x, center_y):
    half_length = COURT_LENGTH_CM / 2.0
    half_width = COURT_WIDTH_CM / 2.0
    return center_x - half_length, center_x + half_length, center_y - half_width, center_y + half_width


def hoop_positions(center_x, center_y):
    left, right, _, _ = court_bounds(center_x, center_y)
    return (left + HOOP_OFFSET_FROM_ENDLINE_CM, center_y), (right - HOOP_OFFSET_FROM_ENDLINE_CM, center_y)


def _arc_points(center, radius, start_deg, end_deg, steps=48):
    start = math.radians(start_deg)
    end = math.radians(end_deg)
    return [
        (
            center[0] + math.cos(start + (end - start) * i / steps) * radius,
            center[1] + math.sin(start + (end - start) * i / steps) * radius,
        )
        for i in range(steps + 1)
    ]


def _polyline_2d(surface, world_to_screen, points, color, width=2):
    screen_points = [world_to_screen(x, y) for x, y in points]
    if len(screen_points) >= 2:
        pygame.draw.lines(surface, color, False, screen_points, width)


def draw_court_2d(surface, world_to_screen, center_x, center_y):
    left, right, top, bottom = court_bounds(center_x, center_y)
    surface.fill((16, 20, 24))
    apron = [
        world_to_screen(left - 180, top - 180),
        world_to_screen(right + 180, top - 180),
        world_to_screen(right + 180, bottom + 180),
        world_to_screen(left - 180, bottom + 180),
    ]
    pygame.draw.polygon(surface, COURT_APRON, apron)
    court_poly = [
        world_to_screen(left, top),
        world_to_screen(right, top),
        world_to_screen(right, bottom),
        world_to_screen(left, bottom),
    ]
    pygame.draw.polygon(surface, COURT_WOOD, court_poly)

    stripe_count = 12
    stripe_width = COURT_LENGTH_CM / stripe_count
    for index in range(stripe_count):
        if index % 2 == 0:
            x0 = left + index * stripe_width
            x1 = x0 + stripe_width
            stripe = [
                world_to_screen(x0, top),
                world_to_screen(x1, top),
                world_to_screen(x1, bottom),
                world_to_screen(x0, bottom),
            ]
            pygame.draw.polygon(surface, COURT_WOOD_ALT, stripe)

    key_top = center_y - KEY_WIDTH_CM / 2.0
    key_bottom = center_y + KEY_WIDTH_CM / 2.0
    left_hoop, right_hoop = hoop_positions(center_x, center_y)
    left_key_end = left + KEY_LENGTH_CM
    right_key_start = right - KEY_LENGTH_CM
    pygame.draw.polygon(surface, COURT_PAINT, [world_to_screen(left, key_top), world_to_screen(left_key_end, key_top), world_to_screen(left_key_end, key_bottom), world_to_screen(left, key_bottom)])
    pygame.draw.polygon(surface, COURT_PAINT, [world_to_screen(right_key_start, key_top), world_to_screen(right, key_top), world_to_screen(right, key_bottom), world_to_screen(right_key_start, key_bottom)])
    _polyline_2d(surface, world_to_screen, [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)], COURT_LINES, 3)
    _polyline_2d(surface, world_to_screen, [(center_x, top), (center_x, bottom)], COURT_LINES, 3)
    _polyline_2d(surface, world_to_screen, _arc_points((center_x, center_y), CENTER_CIRCLE_RADIUS_CM, 0, 360, 64), COURT_LINES, 3)
    _polyline_2d(surface, world_to_screen, _arc_points((center_x, center_y), CENTER_CIRCLE_RADIUS_CM * 0.38, 0, 360, 48), COURT_CENTER_LOGO, 5)

    for hoop_x, side in ((left_hoop[0], "left"), (right_hoop[0], "right")):
        backboard_x = left + BACKBOARD_OFFSET_FROM_ENDLINE_CM if side == "left" else right - BACKBOARD_OFFSET_FROM_ENDLINE_CM
        _polyline_2d(surface, world_to_screen, [(backboard_x, center_y - BACKBOARD_WIDTH_CM / 2.0), (backboard_x, center_y + BACKBOARD_WIDTH_CM / 2.0)], COURT_LINES, 3)
        _polyline_2d(surface, world_to_screen, _arc_points((hoop_x, center_y), 23.0, 0, 360, 24), COURT_LINES, 2)
        if side == "left":
            _polyline_2d(surface, world_to_screen, [(left, key_top), (left_key_end, key_top), (left_key_end, key_bottom), (left, key_bottom)], COURT_LINES, 3)
            _polyline_2d(surface, world_to_screen, _arc_points((left_key_end, center_y), FREE_THROW_RADIUS_CM, -90, 90, 32), COURT_LINES, 3)
            _polyline_2d(surface, world_to_screen, _arc_points((hoop_x, center_y), RESTRICTED_RADIUS_CM, -90, 90, 28), COURT_LINES, 2)
            _polyline_2d(surface, world_to_screen, [(left, top + CORNER_THREE_OFFSET_CM), (left + 280, top + CORNER_THREE_OFFSET_CM)], COURT_LINES, 3)
            _polyline_2d(surface, world_to_screen, [(left, bottom - CORNER_THREE_OFFSET_CM), (left + 280, bottom - CORNER_THREE_OFFSET_CM)], COURT_LINES, 3)
            _polyline_2d(surface, world_to_screen, _arc_points((hoop_x, center_y), THREE_POINT_RADIUS_CM, -68, 68, 56), COURT_LINES, 3)
        else:
            _polyline_2d(surface, world_to_screen, [(right_key_start, key_top), (right, key_top), (right, key_bottom), (right_key_start, key_bottom)], COURT_LINES, 3)
            _polyline_2d(surface, world_to_screen, _arc_points((right_key_start, center_y), FREE_THROW_RADIUS_CM, 90, 270, 32), COURT_LINES, 3)
            _polyline_2d(surface, world_to_screen, _arc_points((hoop_x, center_y), RESTRICTED_RADIUS_CM, 90, 270, 28), COURT_LINES, 2)
            _polyline_2d(surface, world_to_screen, [(right - 280, top + CORNER_THREE_OFFSET_CM), (right, top + CORNER_THREE_OFFSET_CM)], COURT_LINES, 3)
            _polyline_2d(surface, world_to_screen, [(right - 280, bottom - CORNER_THREE_OFFSET_CM), (right, bottom - CORNER_THREE_OFFSET_CM)], COURT_LINES, 3)
            _polyline_2d(surface, world_to_screen, _arc_points((hoop_x, center_y), THREE_POINT_RADIUS_CM, 112, 248, 56), COURT_LINES, 3)


def _projected(camera, focal, points):
    projected = []
    for x, y in points:
        pr = camera.project((x, y, 0.0), focal)
        if pr is None:
            return None
        projected.append((pr[0], pr[1]))
    return projected


def _draw_polyline_3d(surface, camera, focal, points, color, width=2):
    projected = _projected(camera, focal, points)
    if projected and len(projected) >= 2:
        pygame.draw.lines(surface, color, False, projected, width)


def _draw_polygon_3d(surface, camera, focal, points, color):
    projected = _projected(camera, focal, points)
    if projected and len(projected) >= 3:
        pygame.draw.polygon(surface, color, projected)


def draw_court_3d(surface, camera, focal, center_x, center_y):
    left, right, top, bottom = court_bounds(center_x, center_y)
    _draw_polygon_3d(surface, camera, focal, [(left - 180, top - 180), (right + 180, top - 180), (right + 180, bottom + 180), (left - 180, bottom + 180)], COURT_APRON)
    stripe_count = 12
    stripe_width = COURT_LENGTH_CM / stripe_count
    for index in range(stripe_count):
        x0 = left + index * stripe_width
        x1 = x0 + stripe_width
        color = COURT_WOOD_ALT if index % 2 == 0 else COURT_WOOD
        _draw_polygon_3d(surface, camera, focal, [(x0, top), (x1, top), (x1, bottom), (x0, bottom)], color)

    key_top = center_y - KEY_WIDTH_CM / 2.0
    key_bottom = center_y + KEY_WIDTH_CM / 2.0
    left_hoop, right_hoop = hoop_positions(center_x, center_y)
    left_key_end = left + KEY_LENGTH_CM
    right_key_start = right - KEY_LENGTH_CM
    _draw_polygon_3d(surface, camera, focal, [(left, key_top), (left_key_end, key_top), (left_key_end, key_bottom), (left, key_bottom)], COURT_PAINT)
    _draw_polygon_3d(surface, camera, focal, [(right_key_start, key_top), (right, key_top), (right, key_bottom), (right_key_start, key_bottom)], COURT_PAINT)
    _draw_polyline_3d(surface, camera, focal, [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)], COURT_LINES, 3)
    _draw_polyline_3d(surface, camera, focal, [(center_x, top), (center_x, bottom)], COURT_LINES, 3)
    _draw_polyline_3d(surface, camera, focal, _arc_points((center_x, center_y), CENTER_CIRCLE_RADIUS_CM, 0, 360, 64), COURT_LINES, 3)
    _draw_polyline_3d(surface, camera, focal, _arc_points((center_x, center_y), CENTER_CIRCLE_RADIUS_CM * 0.38, 0, 360, 48), COURT_CENTER_LOGO, 4)
    for hoop_x, side in ((left_hoop[0], "left"), (right_hoop[0], "right")):
        backboard_x = left + BACKBOARD_OFFSET_FROM_ENDLINE_CM if side == "left" else right - BACKBOARD_OFFSET_FROM_ENDLINE_CM
        _draw_polyline_3d(surface, camera, focal, [(backboard_x, center_y - BACKBOARD_WIDTH_CM / 2.0), (backboard_x, center_y + BACKBOARD_WIDTH_CM / 2.0)], COURT_LINES, 3)
        _draw_polyline_3d(surface, camera, focal, _arc_points((hoop_x, center_y), 23.0, 0, 360, 24), COURT_LINES, 2)
        if side == "left":
            _draw_polyline_3d(surface, camera, focal, [(left, key_top), (left_key_end, key_top), (left_key_end, key_bottom), (left, key_bottom)], COURT_LINES, 3)
            _draw_polyline_3d(surface, camera, focal, _arc_points((left_key_end, center_y), FREE_THROW_RADIUS_CM, -90, 90, 32), COURT_LINES, 3)
            _draw_polyline_3d(surface, camera, focal, _arc_points((hoop_x, center_y), RESTRICTED_RADIUS_CM, -90, 90, 28), COURT_LINES, 2)
            _draw_polyline_3d(surface, camera, focal, [(left, top + CORNER_THREE_OFFSET_CM), (left + 280, top + CORNER_THREE_OFFSET_CM)], COURT_LINES, 3)
            _draw_polyline_3d(surface, camera, focal, [(left, bottom - CORNER_THREE_OFFSET_CM), (left + 280, bottom - CORNER_THREE_OFFSET_CM)], COURT_LINES, 3)
            _draw_polyline_3d(surface, camera, focal, _arc_points((hoop_x, center_y), THREE_POINT_RADIUS_CM, -68, 68, 56), COURT_LINES, 3)
        else:
            _draw_polyline_3d(surface, camera, focal, [(right_key_start, key_top), (right, key_top), (right, key_bottom), (right_key_start, key_bottom)], COURT_LINES, 3)
            _draw_polyline_3d(surface, camera, focal, _arc_points((right_key_start, center_y), FREE_THROW_RADIUS_CM, 90, 270, 32), COURT_LINES, 3)
            _draw_polyline_3d(surface, camera, focal, _arc_points((hoop_x, center_y), RESTRICTED_RADIUS_CM, 90, 270, 28), COURT_LINES, 2)
            _draw_polyline_3d(surface, camera, focal, [(right - 280, top + CORNER_THREE_OFFSET_CM), (right, top + CORNER_THREE_OFFSET_CM)], COURT_LINES, 3)
            _draw_polyline_3d(surface, camera, focal, [(right - 280, bottom - CORNER_THREE_OFFSET_CM), (right, bottom - CORNER_THREE_OFFSET_CM)], COURT_LINES, 3)
            _draw_polyline_3d(surface, camera, focal, _arc_points((hoop_x, center_y), THREE_POINT_RADIUS_CM, 112, 248, 56), COURT_LINES, 3)
