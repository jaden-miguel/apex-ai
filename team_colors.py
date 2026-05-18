"""F1 team colors – aligned with the *official* Formula 1 brand palette
(2024-2025 season as published on formula1.com, plus reasonable
approximations for the 2026 new entries until their final liveries are
released).  Keeping these on-brand makes the leaderboard, podium card,
and driver dots read instantly to anyone who follows the sport."""

TEAM_COLORS = {
    # Official 2025 F1 grid colours
    "Red Bull Racing": "#3671C6",   # Red Bull Racing Honda RBPT
    "Racing Bulls":    "#6692FF",   # VCARB
    "Ferrari":         "#E80020",   # Scuderia Ferrari Rosso
    "Mercedes":        "#27F4D2",   # Petronas teal
    "McLaren":         "#FF8000",   # Papaya orange
    "Aston Martin":    "#229971",   # British racing green
    "Alpine":          "#00A1E8",   # Alpine blue (BWT)
    "Haas F1 Team":    "#B6BABD",   # Haas titanium grey
    "Williams":        "#64C4FF",   # Williams Atlassian blue
    # 2026 new / re-branded teams (best estimate of unveiled liveries)
    "Audi":            "#00FFCC",   # Audi F1 launch teal
    "Cadillac":        "#003A70",   # Cadillac heritage navy
    # Historical names still present in data.csv
    "RB":              "#6692FF",
    "AlphaTauri":      "#6692FF",
    "Kick Sauber":     "#52E252",   # Stake F1 / Kick Sauber lime
    "Alfa Romeo":      "#900000",
}
