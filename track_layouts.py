"""
Hand-crafted F1 circuit outlines.  Each track is a closed loop of (x, y)
control points that captures the *real* silhouette of the circuit — the
distinctive features that make Suzuka a figure-8, Baku a long sea-front
ribbon, Spa a triangular Ardennes climb, etc.

Coordinates are authored in their natural aspect ratio (no longer forced
into a unit square).  The renderer in `app.py` computes each track's
bounding box at draw time and scales it to fit the canvas while
preserving aspect ratio, so a long-and-skinny circuit no longer gets
stretched into a generic blob.

Coordinates are interpolated into smooth curves at render time.
"""

# ---------------------------------------------------------------------------
# Each circuit below is a closed-loop sequence of (x, y) points traced
# clockwise.  The sequence does *not* duplicate the start point at the
# end — `_interpolate_track` closes the loop.
#
# These shapes are stylised — they're not GPS-accurate, but the
# silhouette (number and orientation of major straights, location of
# the iconic hairpins / chicanes / esses) matches the real layout so
# each track is unambiguous from a glance.
# ---------------------------------------------------------------------------

# Albert Park, Melbourne — flat circuit hugging the lake's perimeter.
# Iconic features: fast Turn 1 sweep at the north-east, the Lakeside
# Drive sweep, then the slow south-east chicane and the long Turn 11
# left-hander coming back along the western side.
MELBOURNE = [
    (0.14, 0.28),                                              # S/F top-left
    (0.30, 0.22), (0.46, 0.20), (0.60, 0.22),                  # north straight
    (0.74, 0.26), (0.86, 0.32), (0.94, 0.42),                  # T1 sweep right
    (0.96, 0.54), (0.92, 0.62),                                # T3-T4
    (0.84, 0.66), (0.74, 0.70), (0.66, 0.66),                  # T5-T6 chicane
    (0.62, 0.72), (0.66, 0.78),                                # T7
    (0.62, 0.84), (0.52, 0.88), (0.40, 0.90),                  # Lakeside Drive
    (0.28, 0.86), (0.18, 0.80),                                # T9 sweep
    (0.10, 0.70), (0.08, 0.58),                                # T10
    (0.10, 0.46),                                              # T11 long left
    (0.14, 0.36),
]

# Sakhir, Bahrain — bowtie with the long Turn 4 sweeper, snaking middle
# sector and the slow Turn 8 hairpin halfway round.
BAHRAIN = [
    (0.18, 0.42), (0.40, 0.42), (0.60, 0.42), (0.78, 0.42),
    (0.88, 0.42), (0.94, 0.46), (0.96, 0.54), (0.92, 0.60),
    (0.84, 0.62), (0.76, 0.58), (0.70, 0.62), (0.66, 0.70),
    (0.58, 0.76), (0.46, 0.80), (0.34, 0.84), (0.22, 0.84),
    (0.14, 0.78), (0.18, 0.70), (0.28, 0.66), (0.36, 0.62),
    (0.42, 0.66), (0.48, 0.64), (0.46, 0.58), (0.38, 0.56),
    (0.30, 0.54), (0.22, 0.50),
]

# Jeddah Corniche — extremely long, narrow, fast street circuit with a
# series of high-speed esses along the seafront.
JEDDAH = [
    (0.05, 0.50), (0.12, 0.46), (0.20, 0.50), (0.28, 0.46),
    (0.36, 0.50), (0.44, 0.46), (0.52, 0.50), (0.60, 0.46),
    (0.68, 0.50), (0.76, 0.46), (0.84, 0.50), (0.92, 0.48),
    (0.97, 0.54), (0.94, 0.62), (0.86, 0.66), (0.80, 0.62),
    (0.72, 0.66), (0.64, 0.62), (0.56, 0.66), (0.48, 0.62),
    (0.40, 0.66), (0.32, 0.62), (0.24, 0.66), (0.16, 0.62),
    (0.10, 0.58),
]

# Suzuka, Japan — the only true figure-8 on the F1 calendar.  The path
# crosses itself near the centre, mimicking the over-bridge between the
# Degner curves and the back straight (the two passes are offset along
# the y axis so the crossover is unambiguous after interpolation).
SUZUKA = [
    # Upper loop, clockwise from S/F at the top
    (0.50, 0.08), (0.62, 0.10), (0.74, 0.16), (0.80, 0.26),
    (0.76, 0.34), (0.66, 0.38),                # 1st Degner area
    (0.56, 0.42),                               # heading toward bridge
    (0.46, 0.46),                               # 1st pass (going down-left)
    # Lower loop continues counter-clockwise
    (0.34, 0.52), (0.22, 0.58), (0.12, 0.66),
    (0.10, 0.76), (0.18, 0.84), (0.32, 0.86),  # hairpin lobe (Hairpin / 200R)
    (0.46, 0.86), (0.58, 0.84), (0.70, 0.80),  # Spoon
    (0.78, 0.74),                               # back straight starts climbing
    (0.70, 0.66), (0.62, 0.58),                # rising back across the centre
    (0.54, 0.50),                               # 2nd pass (crosses over the 1st)
    # Continue around and close the upper loop
    (0.46, 0.40),                               # post-crossover heading up
    (0.40, 0.32), (0.30, 0.28), (0.22, 0.22),
    (0.24, 0.14), (0.34, 0.10),
]

# Shanghai — the famous "snail" coiled inside a triangle.  The opening
# turn is a long descending right-hander that wraps almost 360°.
SHANGHAI = [
    (0.55, 0.10), (0.66, 0.12), (0.76, 0.16), (0.84, 0.22),
    (0.90, 0.30), (0.92, 0.40), (0.88, 0.46), (0.78, 0.48),
    (0.68, 0.46), (0.62, 0.40), (0.60, 0.34), (0.58, 0.28),
    (0.52, 0.30), (0.48, 0.36), (0.50, 0.44), (0.56, 0.50),
    (0.64, 0.56), (0.72, 0.62), (0.80, 0.68), (0.86, 0.76),
    (0.84, 0.84), (0.74, 0.88), (0.62, 0.90), (0.48, 0.90),
    (0.34, 0.88), (0.22, 0.82), (0.14, 0.72), (0.12, 0.60),
    (0.16, 0.48), (0.22, 0.38), (0.30, 0.30), (0.38, 0.22),
    (0.46, 0.16),
]

# Miami International Autodrome — squared-off ring around the Hard Rock
# Stadium with chicanes in the middle sector.
MIAMI = [
    (0.30, 0.12), (0.50, 0.10), (0.70, 0.12), (0.84, 0.18),
    (0.92, 0.30), (0.94, 0.46), (0.88, 0.54), (0.78, 0.54),
    (0.72, 0.50), (0.66, 0.54), (0.70, 0.62), (0.78, 0.66),
    (0.86, 0.72), (0.90, 0.82), (0.82, 0.88), (0.68, 0.90),
    (0.50, 0.92), (0.32, 0.90), (0.18, 0.84), (0.10, 0.74),
    (0.08, 0.62), (0.10, 0.50), (0.14, 0.38), (0.18, 0.28),
    (0.22, 0.20),
]

# Imola — long, narrow, north-south circuit with the Variante chicanes
# and the Tamburello/Villeneuve curves.
IMOLA = [
    (0.45, 0.06), (0.55, 0.06), (0.62, 0.10), (0.66, 0.18),
    (0.62, 0.26), (0.58, 0.32), (0.62, 0.38),  # Acque Minerali
    (0.66, 0.46), (0.62, 0.54), (0.58, 0.62), (0.62, 0.68),
    (0.66, 0.76), (0.62, 0.84), (0.54, 0.90), (0.46, 0.92),
    (0.40, 0.86), (0.42, 0.78), (0.46, 0.70), (0.42, 0.64),
    (0.38, 0.56), (0.42, 0.48), (0.46, 0.40), (0.42, 0.32),
    (0.38, 0.24), (0.40, 0.16), (0.44, 0.10),
]

# Monaco — tight street rectangle around the harbour.  Distinctive
# tunnel curve, swimming-pool chicane and Rascasse hairpin.
MONACO = [
    (0.20, 0.30), (0.32, 0.28), (0.46, 0.28), (0.60, 0.28),  # main straight to Sainte Devote
    (0.74, 0.30), (0.84, 0.34), (0.90, 0.42), (0.92, 0.52),  # Massenet → Casino
    (0.86, 0.56), (0.78, 0.54),                                # Mirabeau
    (0.74, 0.60), (0.78, 0.66),                                # Loews hairpin
    (0.84, 0.70), (0.92, 0.72), (0.94, 0.78),                  # tunnel exit
    (0.88, 0.84), (0.78, 0.86),                                # Nouvelle Chicane
    (0.66, 0.88), (0.54, 0.84),                                # Tabac
    (0.42, 0.80), (0.30, 0.78),                                # Swimming Pool
    (0.20, 0.74), (0.12, 0.66), (0.10, 0.56),                  # Rascasse
    (0.12, 0.46), (0.16, 0.38),
]

# Barcelona-Catalunya — square-ish with the long sweeper into the
# stadium section and the final chicane.
BARCELONA = [
    (0.30, 0.12), (0.50, 0.10), (0.68, 0.12), (0.80, 0.18),
    (0.88, 0.28), (0.92, 0.40), (0.88, 0.50), (0.80, 0.54),
    (0.72, 0.50), (0.66, 0.54), (0.66, 0.62), (0.72, 0.68),
    (0.78, 0.76), (0.80, 0.84), (0.74, 0.90), (0.62, 0.92),
    (0.50, 0.92), (0.38, 0.92), (0.28, 0.88), (0.20, 0.82),
    (0.16, 0.74), (0.20, 0.66), (0.26, 0.62), (0.22, 0.54),
    (0.16, 0.46), (0.14, 0.36), (0.16, 0.26), (0.22, 0.18),
]

# Circuit Gilles Villeneuve, Montreal — long narrow ribbon on Île
# Notre-Dame with the famous "Wall of Champions" chicane near the end.
MONTREAL = [
    (0.06, 0.30), (0.18, 0.26), (0.34, 0.24), (0.50, 0.24),
    (0.66, 0.26), (0.78, 0.30), (0.86, 0.36),
    (0.92, 0.40), (0.96, 0.46), (0.92, 0.52),  # hairpin
    (0.86, 0.56), (0.78, 0.58), (0.70, 0.56),
    (0.62, 0.60), (0.66, 0.66), (0.70, 0.70),  # Wall of Champions chicane
    (0.66, 0.74), (0.58, 0.76), (0.46, 0.78),
    (0.32, 0.78), (0.20, 0.76), (0.10, 0.70),
    (0.04, 0.62), (0.02, 0.50), (0.04, 0.40),
]

# Hungaroring — twisty, low-speed "Monaco without walls" with a hairpin
# at Turn 1 and a long sweeping back section.
HUNGARORING = [
    (0.30, 0.10), (0.46, 0.08), (0.62, 0.12), (0.74, 0.20),
    (0.80, 0.30), (0.74, 0.36), (0.66, 0.34),  # Turn 2 sweep
    (0.60, 0.40), (0.62, 0.50), (0.70, 0.54),
    (0.78, 0.58), (0.84, 0.66), (0.86, 0.76),
    (0.80, 0.84), (0.70, 0.88), (0.58, 0.90),
    (0.46, 0.88), (0.36, 0.84), (0.30, 0.78),
    (0.34, 0.70), (0.40, 0.66), (0.36, 0.58),
    (0.28, 0.54), (0.20, 0.48), (0.16, 0.40),
    (0.18, 0.30), (0.22, 0.22), (0.26, 0.16),
]

# Red Bull Ring, Spielberg — short triangular Alpine circuit.  Three
# long straights joined by tight hairpins (Niki Lauda, Remus, Würth).
SPIELBERG = [
    (0.30, 0.20),                                              # S/F bottom-left
    (0.50, 0.14), (0.70, 0.12), (0.82, 0.16),                  # up to T1
    (0.90, 0.24),                                              # Niki Lauda hairpin (sharp)
    (0.84, 0.34), (0.78, 0.42),                                # short straight to T3
    (0.86, 0.52), (0.92, 0.62), (0.94, 0.72),                  # long Remus straight
    (0.86, 0.80),                                              # Remus hairpin
    (0.74, 0.82), (0.60, 0.84), (0.48, 0.84),                  # back along the bottom
    (0.34, 0.82), (0.22, 0.78),                                # Würth hairpin entry
    (0.14, 0.70),                                              # Würth hairpin
    (0.16, 0.58), (0.20, 0.46), (0.22, 0.34),                  # main straight back up
]

# Silverstone — the classic British circuit with the Loop hairpin at
# Becketts/Maggotts and the long Hangar Straight to Stowe.
SILVERSTONE = [
    (0.20, 0.30), (0.32, 0.24), (0.46, 0.22), (0.60, 0.26),  # Abbey → Village
    (0.70, 0.32), (0.78, 0.40), (0.74, 0.46), (0.66, 0.46),  # Loop hairpin
    (0.60, 0.40), (0.54, 0.38),                                # Aintree
    (0.50, 0.44), (0.54, 0.50), (0.62, 0.54),                  # Wellington Straight
    (0.72, 0.58), (0.82, 0.62), (0.90, 0.66),                  # Brooklands
    (0.94, 0.74), (0.90, 0.82), (0.80, 0.88), (0.66, 0.90),   # Stowe → Vale
    (0.52, 0.92), (0.40, 0.90), (0.28, 0.86), (0.18, 0.80),   # Club
    (0.12, 0.70), (0.10, 0.60), (0.12, 0.50),                  # Hangar Straight back
    (0.16, 0.42), (0.18, 0.36),
]

# Spa-Francorchamps — long Ardennes triangle with Eau Rouge, the Kemmel
# Straight, Pouhon double-left and Bus Stop chicane.
SPA = [
    (0.30, 0.92), (0.42, 0.94), (0.54, 0.92), (0.62, 0.86),  # La Source hairpin
    (0.58, 0.78),                                              # down to Eau Rouge
    (0.66, 0.72), (0.74, 0.62),                                # Eau Rouge / Raidillon
    (0.82, 0.50), (0.88, 0.36), (0.92, 0.22), (0.86, 0.12),   # Kemmel + Les Combes
    (0.74, 0.10), (0.62, 0.16), (0.54, 0.24),                  # Malmedy / Rivage
    (0.46, 0.32),                                              # Pouhon double-left
    (0.38, 0.42), (0.30, 0.50), (0.22, 0.58), (0.16, 0.66),   # Fagnes, Stavelot
    (0.12, 0.74), (0.16, 0.80), (0.22, 0.84),                  # Blanchimont
    (0.20, 0.90),                                              # Bus Stop chicane
]

# Zandvoort — short coastal circuit with the famous banked Hugenholtz
# hairpin (Turn 3) and the banked final corner.
ZANDVOORT = [
    (0.30, 0.18), (0.46, 0.14), (0.60, 0.16), (0.72, 0.22),
    (0.80, 0.30), (0.78, 0.40), (0.70, 0.42),                  # Hugenholtzbocht banked T3
    (0.62, 0.40), (0.58, 0.46), (0.62, 0.54), (0.70, 0.58),
    (0.80, 0.60), (0.86, 0.66), (0.88, 0.74), (0.84, 0.82),   # banked Arie Luyendijk
    (0.74, 0.86), (0.62, 0.84), (0.50, 0.80), (0.38, 0.78),
    (0.26, 0.74), (0.18, 0.66), (0.14, 0.56), (0.16, 0.46),
    (0.20, 0.36), (0.24, 0.26),
]

# Monza — the temple of speed.  Distinctive triangle of three long
# straights linked by chicanes, with the iconic sweeping Parabolica
# closing the lap back onto the main straight.
MONZA = [
    (0.10, 0.20),                                              # Variante del Rettifilo
    (0.18, 0.14), (0.28, 0.12),
    (0.32, 0.18), (0.38, 0.14),                                # 1st chicane kink
    (0.50, 0.10), (0.66, 0.08), (0.80, 0.10), (0.90, 0.16),    # main straight
    (0.94, 0.26), (0.92, 0.36),                                # Curva Grande
    (0.86, 0.44), (0.78, 0.46), (0.74, 0.42),                  # 1st Lesmo
    (0.70, 0.48), (0.74, 0.54),                                # Variante della Roggia
    (0.80, 0.58), (0.86, 0.62),                                # 2nd Lesmo
    (0.84, 0.70),                                              # Serraglio
    (0.78, 0.76), (0.70, 0.78),                                # Variante Ascari (R-L-R)
    (0.74, 0.84), (0.80, 0.90),
    (0.74, 0.94), (0.62, 0.92), (0.50, 0.86), (0.38, 0.78),    # Parabolica
    (0.28, 0.66), (0.20, 0.54), (0.14, 0.42), (0.10, 0.32),    # back along pit straight
]

# Baku City Circuit — extremely long L-shaped seafront straight with the
# tight, twisty old-town castle section.
BAKU = [
    (0.05, 0.55), (0.20, 0.55), (0.40, 0.55), (0.60, 0.55),
    (0.78, 0.55), (0.90, 0.55),                                # 2.2km main straight
    (0.96, 0.60), (0.94, 0.68), (0.86, 0.72), (0.76, 0.72),
    (0.68, 0.74), (0.60, 0.78), (0.52, 0.80),
    (0.46, 0.72), (0.50, 0.66), (0.42, 0.62),                  # old-town zigzag
    (0.36, 0.66), (0.32, 0.72), (0.26, 0.74), (0.18, 0.72),
    (0.12, 0.66), (0.08, 0.62),
]

# Marina Bay, Singapore — angular street rectangle with the Anderson
# Bridge twirl and the Marina Bay grandstand corners.
SINGAPORE = [
    (0.20, 0.20), (0.36, 0.16), (0.54, 0.16), (0.70, 0.20),
    (0.82, 0.28), (0.88, 0.38), (0.86, 0.48), (0.80, 0.52),
    (0.74, 0.48), (0.68, 0.52), (0.64, 0.60), (0.70, 0.66),
    (0.78, 0.72), (0.86, 0.78), (0.88, 0.86), (0.80, 0.92),
    (0.66, 0.94), (0.50, 0.94), (0.34, 0.92), (0.22, 0.86),
    (0.14, 0.78), (0.10, 0.68), (0.08, 0.58), (0.10, 0.48),
    (0.14, 0.38), (0.16, 0.30),
]

# Circuit of the Americas, Austin — counter-clockwise.  Distinctive
# silhouette: the iconic uphill Turn 1 at the west, the snake esses
# along the top, the long left-hander dipping south to the T11 hairpin,
# back straight east, and the stadium zigzag returning to S/F.
AUSTIN = [
    (0.18, 0.42),                                              # S/F (pit straight)
    (0.18, 0.32), (0.22, 0.22), (0.30, 0.14),                  # climb to T1
    (0.42, 0.10), (0.54, 0.12),                                # T1 → T2 sweep
    (0.62, 0.18), (0.56, 0.24), (0.62, 0.30),                  # snake esses
    (0.56, 0.36), (0.62, 0.42), (0.56, 0.48),                  # T6-T9 zigzag
    (0.62, 0.54), (0.70, 0.58), (0.78, 0.62),                  # T10 wide left
    (0.84, 0.70), (0.88, 0.80),                                # back straight east
    (0.84, 0.88), (0.74, 0.90), (0.62, 0.86),                  # T11 hairpin (south-east)
    (0.50, 0.82), (0.42, 0.78),                                # back straight west
    (0.36, 0.72), (0.42, 0.66), (0.36, 0.60),                  # stadium chicane
    (0.28, 0.58), (0.22, 0.52),                                # return to S/F
]

# Mexico City — long pit straight, the wide-sweeping Peraltada at the
# end, and the distinctive Foro Sol stadium pinch where the track
# zigzags through the old baseball stadium grandstands.
MEXICO = [
    (0.14, 0.18),                                              # S/F top-left
    (0.30, 0.16), (0.50, 0.14), (0.68, 0.16),                  # 1.2km main straight
    (0.82, 0.22), (0.90, 0.32), (0.92, 0.44),                  # T1-T3 climb / descent
    (0.86, 0.54), (0.78, 0.56), (0.70, 0.50),                  # T4-T6 esses
    (0.66, 0.58), (0.74, 0.66),                                # T7-T8
    (0.84, 0.72), (0.92, 0.78),                                # back loop / Peraltada
    (0.92, 0.88),                                              # Peraltada banking
    (0.84, 0.92), (0.72, 0.92),                                # Foro Sol stadium kink (in)
    (0.66, 0.86), (0.58, 0.84), (0.52, 0.90),                  #     stadium pinch
    (0.46, 0.94), (0.36, 0.92),                                # Foro Sol exit
    (0.30, 0.86), (0.32, 0.78),
    (0.26, 0.70), (0.20, 0.62), (0.16, 0.52),                  # back to pit straight
    (0.12, 0.40), (0.12, 0.28),
]

# Interlagos, São Paulo — short anti-clockwise loop with the iconic
# downhill Senna S, the Descida do Lago and the long uphill back to
# the start/finish.
INTERLAGOS = [
    (0.50, 0.12), (0.62, 0.14), (0.74, 0.20), (0.84, 0.30),  # uphill back to S/F
    (0.88, 0.40), (0.84, 0.48), (0.74, 0.50),                 # Junçao
    (0.66, 0.46), (0.58, 0.42), (0.50, 0.40),                 # Bico de Pato
    (0.42, 0.46), (0.38, 0.54), (0.42, 0.62),                 # Pinheirinho / Mergulho
    (0.50, 0.66), (0.58, 0.72), (0.64, 0.80),                 # Cotovelo / Descida do Lago
    (0.58, 0.86), (0.46, 0.88), (0.34, 0.86),                 # Curva do Lago
    (0.24, 0.80), (0.18, 0.70), (0.16, 0.58),                 # Subida dos Boxes
    (0.20, 0.46), (0.26, 0.36),                                # Senna S
    (0.32, 0.30), (0.40, 0.22),
]

# Las Vegas Strip Circuit — extremely long thin oval with the iconic
# Strip straight along the south and the East Harmon back-straight.
LAS_VEGAS = [
    (0.05, 0.45), (0.20, 0.42), (0.40, 0.40), (0.60, 0.40),
    (0.78, 0.42), (0.90, 0.46), (0.96, 0.52),                  # roundabout sweep
    (0.94, 0.60), (0.84, 0.62), (0.74, 0.62),
    (0.58, 0.64), (0.40, 0.66), (0.22, 0.66),                  # Strip straight back
    (0.10, 0.62), (0.04, 0.56),
]

# Lusail, Qatar — long fast sweepers, "the desert Maggotts/Becketts".
# Distinctive succession of high-speed double-apex curves and a long
# pit straight along the south.
LUSAIL = [
    (0.14, 0.78),                                              # S/F
    (0.32, 0.74), (0.50, 0.72), (0.66, 0.74),                  # main straight
    (0.78, 0.78), (0.86, 0.74),                                # T1 sweep
    (0.84, 0.66), (0.78, 0.60), (0.86, 0.54),                  # T4-T6 esses
    (0.92, 0.46), (0.94, 0.36),                                # T7 long parabolic
    (0.88, 0.26), (0.78, 0.20),                                # T9 carousel
    (0.66, 0.18), (0.54, 0.14),                                # T10-T12 sweepers
    (0.42, 0.16), (0.32, 0.22),                                # T13-T14
    (0.24, 0.30), (0.18, 0.40), (0.14, 0.50),                  # T15 long left
    (0.10, 0.62), (0.10, 0.72),                                # back to S/F
]

# Yas Marina, Abu Dhabi — angular layout around the marina with the
# Yas Hotel section.  The track has two long straights joined by a
# tight north-end hairpin and a wide south-side sweeper.
YAS_MARINA = [
    (0.14, 0.30),                                              # S/F top of north straight
    (0.30, 0.22), (0.46, 0.18), (0.60, 0.18),                  # T1 long left
    (0.72, 0.20), (0.80, 0.26), (0.86, 0.34),                  # T5-T6
    (0.82, 0.42), (0.74, 0.42), (0.66, 0.38),                  # T7 chicane (right-left)
    (0.60, 0.44), (0.66, 0.50),                                # T8 chicane (left-right)
    (0.74, 0.54), (0.84, 0.58), (0.92, 0.66),                  # back straight to T11
    (0.94, 0.76),                                              # T11 hairpin
    (0.86, 0.84),                                              # under-hotel section
    (0.74, 0.86), (0.62, 0.88), (0.50, 0.86),                  # marina chicane
    (0.40, 0.82), (0.42, 0.72),                                # T16
    (0.34, 0.66), (0.22, 0.62),                                # T18 west
    (0.14, 0.54), (0.10, 0.44),                                # T20 long left back to S/F
]

# Fallback for anything we don't recognise.
GENERIC = [
    (0.50, 0.10), (0.62, 0.12), (0.73, 0.18), (0.82, 0.28),
    (0.88, 0.40), (0.90, 0.52), (0.88, 0.64), (0.82, 0.74),
    (0.73, 0.84), (0.62, 0.90), (0.50, 0.92), (0.38, 0.90),
    (0.27, 0.84), (0.18, 0.74), (0.12, 0.62), (0.10, 0.50),
    (0.12, 0.38), (0.18, 0.28), (0.27, 0.18), (0.38, 0.12),
]


def _keywords(name):
    """Lowercase keyword tokens from an event name."""
    return name.lower().replace("grand prix", "gp").split()


_TRACK_MAP = {
    "australia": MELBOURNE, "melbourne": MELBOURNE, "australian": MELBOURNE,
    "bahrain": BAHRAIN, "sakhir": BAHRAIN,
    "saudi": JEDDAH, "jeddah": JEDDAH, "arabia": JEDDAH,
    "japan": SUZUKA, "suzuka": SUZUKA, "japanese": SUZUKA,
    "china": SHANGHAI, "shanghai": SHANGHAI, "chinese": SHANGHAI,
    "miami": MIAMI,
    "emilia": IMOLA, "imola": IMOLA, "romagna": IMOLA,
    "monaco": MONACO,
    "spain": BARCELONA, "barcelona": BARCELONA, "spanish": BARCELONA, "catalunya": BARCELONA,
    "canada": MONTREAL, "montreal": MONTREAL, "canadian": MONTREAL,
    "hungary": HUNGARORING, "hungaroring": HUNGARORING, "hungarian": HUNGARORING,
    "austria": SPIELBERG, "spielberg": SPIELBERG, "austrian": SPIELBERG,
    "britain": SILVERSTONE, "silverstone": SILVERSTONE, "british": SILVERSTONE,
    "belgium": SPA, "spa": SPA, "belgian": SPA,
    "netherlands": ZANDVOORT, "zandvoort": ZANDVOORT, "dutch": ZANDVOORT,
    "italy": MONZA, "monza": MONZA, "italian": MONZA,
    "azerbaijan": BAKU, "baku": BAKU,
    "singapore": SINGAPORE, "marina": SINGAPORE,
    "united states": AUSTIN, "austin": AUSTIN, "cota": AUSTIN,
    "mexico": MEXICO, "mexican": MEXICO,
    "brazil": INTERLAGOS, "interlagos": INTERLAGOS, "são paulo": INTERLAGOS, "sao paulo": INTERLAGOS, "brazilian": INTERLAGOS,
    "las vegas": LAS_VEGAS, "vegas": LAS_VEGAS,
    "qatar": LUSAIL, "lusail": LUSAIL,
    "abu dhabi": YAS_MARINA, "yas": YAS_MARINA, "dhabi": YAS_MARINA,
}

# Human-friendly names for display
_NAME_MAP = {
    id(MELBOURNE): "Albert Park", id(BAHRAIN): "Sakhir",
    id(JEDDAH): "Jeddah Corniche", id(SUZUKA): "Suzuka",
    id(SHANGHAI): "Shanghai", id(MIAMI): "Miami Autodrome",
    id(IMOLA): "Imola", id(MONACO): "Monaco",
    id(BARCELONA): "Barcelona-Catalunya", id(MONTREAL): "Circuit Gilles Villeneuve",
    id(HUNGARORING): "Hungaroring",
    id(SPIELBERG): "Red Bull Ring", id(SILVERSTONE): "Silverstone",
    id(SPA): "Spa-Francorchamps", id(ZANDVOORT): "Zandvoort",
    id(MONZA): "Monza", id(BAKU): "Baku City Circuit",
    id(SINGAPORE): "Marina Bay", id(AUSTIN): "COTA",
    id(MEXICO): "Autódromo Hermanos Rodríguez",
    id(INTERLAGOS): "Interlagos", id(LAS_VEGAS): "Las Vegas Strip",
    id(LUSAIL): "Lusail", id(YAS_MARINA): "Yas Marina",
    id(GENERIC): "Circuit",
}


def get_track(race_name: str):
    """Return (points, circuit_name) for a race name string."""
    low = race_name.lower()
    for key, layout in _TRACK_MAP.items():
        if key in low:
            return layout, _NAME_MAP.get(id(layout), "Circuit")
    return GENERIC, _NAME_MAP.get(id(GENERIC), "Circuit")
