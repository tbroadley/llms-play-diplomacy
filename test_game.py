from game import DiplomacyGame


def test_game_get_current_state():
    game = DiplomacyGame(turn_time_limit=0)
    assert game.get_current_state("AUSTRIA") == """
You are playing as AUSTRIA in SPRING 1901 MOVEMENT.
You control the following units:
- A BUD
- A VIE
- F TRI

Supply centers owned: BUD, TRI, VIE

Current state of all units on the board:
- AUSTRIA: A BUD (can move to GAL, RUM, SER, TRI, VIE)
- AUSTRIA: A VIE (can move to BOH, BUD, GAL, TRI, TYR)
- AUSTRIA: F TRI (can move to ADR, ALB, BUD, SER, TYR, VEN, VIE)
- ENGLAND: F EDI (can move to CLY, lvp, NTH, NWG, YOR)
- ENGLAND: F LON (can move to ENG, NTH, YOR, WAL)
- ENGLAND: A LVP (can move to CLY, edi, IRI, NAO, WAL, yor)
- FRANCE: F BRE (can move to ENG, GAS, MAO, PAR, PIC)
- FRANCE: A MAR (can move to BUR, gas, LYO, PIE, SPA/SC, SWI)
- FRANCE: A PAR (can move to BUR, BRE, GAS, PIC)
- GERMANY: F KIE (can move to BAL, BER, DEN, HEL, HOL, MUN, RUH)
- GERMANY: A BER (can move to BAL, KIE, MUN, PRU, SIL)
- GERMANY: A MUN (can move to BER, BOH, BUR, KIE, RUH, SIL, TYR, SWI)
- ITALY: F NAP (can move to APU, ION, ROM, TYS)
- ITALY: A ROM (can move to apu, NAP, TUS, TYS, ven)
- ITALY: A VEN (can move to ADR, APU, pie, rom, TRI, tus, TYR)
- RUSSIA: A WAR (can move to GAL, LVN, MOS, PRU, SIL, UKR)
- RUSSIA: A MOS (can move to LVN, SEV, STP, UKR, WAR)
- RUSSIA: F SEV (can move to ARM, BLA, MOS, RUM, UKR)
- RUSSIA: F STP/SC (can move to BOT, FIN, LVN)
- TURKEY: F ANK (can move to ARM, BLA, CON, smy)
- TURKEY: A CON (can move to AEG, BUL/EC, BUL/SC, BLA, ANK, SMY)
- TURKEY: A SMY (can move to AEG, ank, arm, CON, EAS, SYR)

Other powers' supply centers:
- AUSTRIA: BUD, TRI, VIE
- ENGLAND: EDI, LON, LVP
- FRANCE: BRE, MAR, PAR
- GERMANY: BER, KIE, MUN
- ITALY: NAP, ROM, VEN
- RUSSIA: MOS, SEV, STP, WAR
- TURKEY: ANK, CON, SMY

Public messages:


Private messages:

"""
