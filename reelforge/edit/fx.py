"""Cinematic visual effects: vignette, subtle film grain, and contrast curves."""


def generate_cinematic_fx_filter(vignette: bool = True, film_grain: bool = True) -> str:
    """Generate subtle aesthetic enhancement filter chain for modern short-form video."""
    filters = []
    if vignette:
        # Subtle dark vignette border
        filters.append("vignette=PI/5")
    if film_grain:
        # Subtle analog grain texture
        filters.append("noise=alls=8:allf=t+u")
    return ",".join(filters)
