# Example synthetic region sets for demo purposes.
PRAGUE = ["CZ0100"]
BIG_CITIES = ["CZ0201", "CZ0202", "CZ0401"]
RURAL = ["CZ0311", "CZ0312", "CZ0321", "CZ0322"]


def choose_regions_for_profile(profile: str):
    if profile == "urban_late_55":
        return RURAL + BIG_CITIES[:1]
    if profile == "rural_first_30":
        return RURAL
    if profile == "balanced_70":
        return RURAL + BIG_CITIES + PRAGUE
    return RURAL
