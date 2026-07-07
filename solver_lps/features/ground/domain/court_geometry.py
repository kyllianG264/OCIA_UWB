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
