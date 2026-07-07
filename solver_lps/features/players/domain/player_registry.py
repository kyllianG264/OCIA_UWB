from solver_lps.features.players.domain.player_analytics import (
    capture_heatmap_snapshot,
    create_player_analytics,
    reset_player_analytics,
    toggle_player_card,
    update_player_analytics,
)


DEFAULT_PLAYER_ID = "tag:uwb"


def create_player_registry(center_x, center_y):
    return {
        "center_x": center_x,
        "center_y": center_y,
        "bounds": None,
        "selected_player_id": DEFAULT_PLAYER_ID,
        "profiles": {},
    }


def ensure_player_registry_defaults(registry):
    registry.setdefault("center_x", 1250.0)
    registry.setdefault("center_y", 900.0)
    registry.setdefault("bounds", None)
    registry.setdefault("selected_player_id", DEFAULT_PLAYER_ID)
    registry.setdefault("profiles", {})
    return registry


def reset_player_registry(registry):
    ensure_player_registry_defaults(registry)
    selected_player_id = registry.get("selected_player_id", DEFAULT_PLAYER_ID)
    for profile in registry["profiles"].values():
        reset_player_analytics(profile)
    registry["selected_player_id"] = selected_player_id


def ensure_player_profile(registry, player_id, *, name=None, source_label="Estimation UWB"):
    ensure_player_registry_defaults(registry)
    profiles = registry["profiles"]
    if player_id not in profiles:
        profiles[player_id] = create_player_analytics(
            registry["center_x"],
            registry["center_y"],
            bounds=registry.get("bounds"),
        )
        profiles[player_id]["player_id"] = player_id
    profile = profiles[player_id]
    if name is not None:
        profile["name"] = name
    if source_label is not None:
        profile["source_label"] = source_label
    return profile


def select_player_profile(registry, player_id, *, show_card=None):
    profile = ensure_player_profile(
        registry,
        player_id,
        name=_default_player_name(player_id),
    )
    registry["selected_player_id"] = player_id
    if show_card is not None:
        profile["card_visible"] = bool(show_card)
    return profile


def get_selected_player_profile(registry):
    ensure_player_registry_defaults(registry)
    selected_player_id = registry.get("selected_player_id") or DEFAULT_PLAYER_ID
    return ensure_player_profile(registry, selected_player_id, name=_default_player_name(selected_player_id))


def update_registry_player(
    registry,
    player_id,
    *,
    t,
    pos_xy,
    height_cm,
    jump_extra_cm,
    dt,
    name=None,
    source_label="Estimation UWB",
    heatmap_pos_xy=None,
):
    profile = ensure_player_profile(
        registry,
        player_id,
        name=name or _default_player_name(player_id),
        source_label=source_label,
    )
    update_player_analytics(
        profile,
        t,
        pos_xy,
        height_cm,
        jump_extra_cm,
        dt,
        heatmap_pos_xy=heatmap_pos_xy,
    )
    return profile


def update_registry_players_from_positions(
    registry,
    positions,
    *,
    t,
    dt,
    source_label="Tracking CV",
):
    updated_profiles = []
    for player in positions or []:
        player_id = player.get("player_id") or player.get("stable_id")
        if not player_id:
            continue
        player_x = player.get("x_cm", player.get("x", player.get("raw_x")))
        player_y = player.get("y_cm", player.get("y", player.get("raw_y")))
        heatmap_x = player.get("heatmap_x", player.get("x", player_x))
        heatmap_y = player.get("heatmap_y", player.get("y", player_y))
        if player_x is None or player_y is None:
            continue
        updated_profiles.append(
            update_registry_player(
                registry,
                str(player_id),
                t=t,
                pos_xy=(player_x, player_y),
                height_cm=None,
                jump_extra_cm=None,
                dt=dt,
                name=str(player_id),
                source_label=source_label,
                heatmap_pos_xy=(heatmap_x, heatmap_y),
            )
        )
    return updated_profiles


def set_registry_bounds(registry, bounds):
    ensure_player_registry_defaults(registry)
    registry["bounds"] = None if bounds is None else tuple(float(value) for value in bounds)
    if registry["bounds"] is None:
        return
    for profile in registry["profiles"].values():
        profile["bounds"] = registry["bounds"]


def capture_selected_player_heatmap(registry):
    capture_heatmap_snapshot(get_selected_player_profile(registry))


def toggle_selected_player_card(registry):
    toggle_player_card(get_selected_player_profile(registry))


def bind_selected_player_analytics(state):
    registry = state["player_registry"]
    state["player_analytics"] = get_selected_player_profile(registry)
    return state["player_analytics"]


def _default_player_name(player_id):
    if player_id == DEFAULT_PLAYER_ID:
        return "Tag UWB"
    return str(player_id)

