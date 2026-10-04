"""Extract original DECAY1/DECAY2/ADDANG3 pre-cut oracle, validation only."""
from pathlib import Path
import argparse
import hashlib
import json
import re


def extract_two_body_source(source, common):
    """Share the exact pre-cut extraction with the composed RESDEC oracle."""
    def routine(name):
        start = source.index("      SUBROUTINE " + name + "(")
        end = source.index("\n      END\n", start) + len("\n      END\n")
        return source[start:end]

    decays = ""
    for name in ["DECAY1", "DECAY2"]:
        original = routine(name)
        start = original.index("C  CALCULATE AUXILIARY QUANTITIES")
        end = original.index("#if __UPWARD__", start)
        first = original[start:end]
        start = original.index("C  FIRST PRODUCT PARTICLE" if name == "DECAY1" else "C  FIRST MUON")
        end = original.index("#if __UPWARD__", start)
        second = original[start:end]
        pol1 = "      SECPAR(11)=POLART\n      SECPAR(12)=POLARF\n" if name == "DECAY2" else ""
        pol2 = "      SECPAR(11)=-POLART\n      SECPAR(12)=POLARF+PI\n" if name == "DECAY2" else ""
        signature = "(M0,M3,M4)" if name == "DECAY1" else "(M0)"
        decays += "      SUBROUTINE " + name + signature + "\n      IMPLICIT NONE\n" + common + """      DOUBLE PRECISION AUX1,AUX2,AUX2A,AUX3,AUX4,COSTCM,
     * COSTH3,COSTH4,GAMMA3,GAMMA4,PHI4,WORK1,WORK2
      INTEGER M0,M3,M4
""" + ("      M3=5\n      M4=6\n" if name == "DECAY2" else "") + first + """      SECPAR(0)=M4
      SECPAR(1)=GAMMA4
""" + pol1 + "      CALL CAPTURE\n" + second + """      SECPAR(0)=M3
      SECPAR(1)=GAMMA3
""" + pol2 + "      CALL CAPTURE\n      RETURN\n      END\n"
    angle = routine("ADDANG3")
    angle = re.sub(r'#define .*?\n#include "corsika.h"\n', "", angle, count=1, flags=re.S)
    if "#" in decays or "#" in angle:
        raise RuntimeError("Unremoved preprocessor scaffolding")
    return decays + angle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest != "0d52382ad2d604f75ca28195ad0e0f581dde3cc0386ee7ef69bb9045a18bf2fd":
        raise RuntimeError("Changed C7 source; audit extraction again")
    common = """      DOUBLE PRECISION SECPAR(0:16),PAMA(51),RD(4),PI,PI2,
     * POLART,POLARF,FIELDS(6,2),GAMMA,BETA,COSTHE,PHIX,PHIY
      INTEGER RS,NUSED,COUNT,IDS(2)
      COMMON /DSTATE/ SECPAR,PAMA,RD,PI,PI2,POLART,POLARF,
     * FIELDS,GAMMA,BETA,COSTHE,PHIX,PHIY
      COMMON /ISTATE/ RS,NUSED,COUNT,IDS
"""
    decays = extract_two_body_source(source, common)
    driver = """      PROGRAM REFERENCE
      IMPLICIT NONE
""" + common + """      INTEGER N,I,J,K,MODE
      DOUBLE PRECISION TOTAL
      CHARACTER*1024 INPUT,OUTPUT
      CALL GET_COMMAND_ARGUMENT(1,INPUT)
      CALL GET_COMMAND_ARGUMENT(2,OUTPUT)
      OPEN(10,FILE=TRIM(INPUT),STATUS='OLD')
      OPEN(11,FILE=TRIM(OUTPUT),STATUS='REPLACE')
      READ(10,*) N
      DO I=1,N
        PAMA=0.D0
        SECPAR=0.D0
        FIELDS=0.D0
        IDS=0
        READ(10,*) MODE,TOTAL,PAMA(50),PAMA(5),PAMA(6),
     *    PHIX,PHIY,COSTHE,RS
        GAMMA=TOTAL/PAMA(50)
        PAMA=PAMA*0.001D0
        PHIY=-PHIY
        BETA=SQRT((GAMMA-1.D0)*(GAMMA+1.D0))/GAMMA
        PI=ACOS(-1.D0)
        PI2=2.D0*PI
        NUSED=0
        COUNT=0
        IF(MODE.EQ.0) THEN
          CALL DECAY1(50,5,6)
        ELSE
          CALL DECAY2(50)
        ENDIF
        WRITE(11,*) MODE,COUNT,NUSED,RS,IDS,
     *    ((FIELDS(K,J),K=1,6),J=1,2)
      ENDDO
      CLOSE(10)
      CLOSE(11)
      END

      SUBROUTINE CAPTURE
      IMPLICIT NONE
""" + common + """      COUNT=COUNT+1
      IDS(COUNT)=NINT(SECPAR(0))
      IF(PAMA(IDS(COUNT)).EQ.0.D0) THEN
        FIELDS(1,COUNT)=SECPAR(1)*1000.D0
      ELSE
        FIELDS(1,COUNT)=SECPAR(1)*PAMA(IDS(COUNT))*1000.D0
      ENDIF
      FIELDS(2,COUNT)=SECPAR(3)
      FIELDS(3,COUNT)=-SECPAR(4)
      FIELDS(4,COUNT)=SECPAR(2)
      FIELDS(5,COUNT)=SECPAR(11)
      FIELDS(6,COUNT)=SECPAR(12)
      END

      SUBROUTINE RMMARD(RANDOMS,N,ISEQ)
      IMPLICIT NONE
""" + common + """      INTEGER N,ISEQ,J
      DOUBLE PRECISION RANDOMS(N)
      IF(ISEQ.NE.1) STOP 31
      DO J=1,N
        RS=MOD(16807_8*RS,2147483647_8)
        RANDOMS(J)=DBLE(RS)/2147483647.D0
        NUSED=NUSED+1
      ENDDO
      END
"""
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "two_body_oracle.f").write_text(driver + decays)
    (args.output / "two_body_oracle_provenance.json").write_text(json.dumps({
        "source": str(args.source.resolve()), "sha256": digest,
        "routines": ["DECAY1 pre-cut", "DECAY2 pre-cut", "ADDANG3"],
        "not_linked_to_cpp": True,
        "changes": ["Keep original auxiliary, energy, angle and polarization arithmetic",
                    "Replace angular cuts/deposition/TSTACK with pre-cut capture",
                    "Explicit total MeV input converted to original GeV masses; dummy indices 50,5,6",
                    "Park-Miller test stream 1, not production C7 RNG or full-shower reference"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
