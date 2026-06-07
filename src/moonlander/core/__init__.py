"""Pure-stdlib game core (terrain, physics, game loop).

Must stay importable in Pyodide with zero package downloads: only ``math``,
``random`` and ``json`` are allowed here (see docs/DESIGN.md).
"""
