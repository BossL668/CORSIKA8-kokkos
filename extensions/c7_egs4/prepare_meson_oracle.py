"""Separate original C7 RHO0DC/ADDANG3 fixture, never linked to native C++."""
from pathlib import Path
import argparse
import hashlib
import json
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest != "0d52382ad2d604f75ca28195ad0e0f581dde3cc0386ee7ef69bb9045a18bf2fd":
        raise RuntimeError("Changed C7 source; audit extraction again")
    common = """      DOUBLE PRECISION CURPAR(0:16),SECPAR(0:16),PAMA(51),
     * RD(4),PI,PI2,POLART,POLARF,FIELDS(6,2),FIRST
      INTEGER RS,NUSED,COUNT,IDS(2)
      COMMON /DSTATE/ CURPAR,SECPAR,PAMA,RD,PI,PI2,POLART,
     * POLARF,FIELDS,FIRST
      COMMON /ISTATE/ RS,NUSED,COUNT,IDS
"""

    def routine(name):
        start = source.index("      SUBROUTINE " + name + "(")
        end = source.index("\n      END\n", start) + len("\n      END\n")
        return source[start:end]

    rho = routine("RHO0DC")
    start = rho.index("C  ADD RARE DECAY")
    end = rho.index("#if __UPWARD__", start)
    first = rho[start:end]
    start = rho.index("C  SECOND PRODUCT PARTICLE")
    end = rho.index("#if __UPWARD__", start)
    second = rho[start:end]
    rho = "      SUBROUTINE RHO0DC(KFROM)\n      IMPLICIT NONE\n" + common + """      DOUBLE PRECISION AUX2A,BETA,COSTCM,COSTH3,COSTH4,
     * GAMMA3,GAMMA4,PHI4,WORK1,WORK2,PAMSEC
      INTEGER KFROM
""" + first + """      SECPAR(1)=GAMMA4
      IF(SECPAR(0).EQ.6.D0) THEN
        SECPAR(11)=POLART
        SECPAR(12)=POLARF
      ENDIF
      CALL CAPTURE
      SECPAR(11)=0.D0
      SECPAR(12)=0.D0
""" + second + """      SECPAR(1)=GAMMA3
      IF(SECPAR(0).EQ.5.D0) THEN
        SECPAR(11)=-POLART
        SECPAR(12)=POLARF+PI
      ENDIF
      CALL CAPTURE
      RETURN
      END
"""
    angle = routine("ADDANG3")
    angle = re.sub(r'#define .*?\n#include "corsika.h"\n', "", angle, count=1, flags=re.S)
    if "#" in rho or "#" in angle:
        raise RuntimeError("Unremoved preprocessor scaffolding")
    driver = """      PROGRAM REFERENCE
      IMPLICIT NONE
""" + common + """      INTEGER N,I,J,K,ORIGIN
      DOUBLE PRECISION TOTAL
      CHARACTER*1024 INPUT,OUTPUT
      CALL GET_COMMAND_ARGUMENT(1,INPUT)
      CALL GET_COMMAND_ARGUMENT(2,OUTPUT)
      OPEN(10,FILE=TRIM(INPUT),STATUS='OLD')
      OPEN(11,FILE=TRIM(OUTPUT),STATUS='REPLACE')
      READ(10,*) N
      DO I=1,N
        CURPAR=0.D0
        SECPAR=0.D0
        FIELDS=0.D0
        IDS=0
        READ(10,*) ORIGIN,TOTAL,PAMA(51),PAMA(9),PAMA(6),
     *    CURPAR(3),CURPAR(4),CURPAR(2),RS,FIRST
        CURPAR(4)=-CURPAR(4)
        CURPAR(1)=TOTAL/PAMA(51)
        PAMA(8)=PAMA(9)
        PAMA(5)=PAMA(6)
        PI=ACOS(-1.D0)
        PI2=2.D0*PI
        NUSED=0
        COUNT=0
        CALL RHO0DC(ORIGIN)
        WRITE(11,*) ORIGIN,COUNT,NUSED,RS,IDS,
     *    ((FIELDS(K,J),K=1,6),J=1,2)
      ENDDO
      CLOSE(10)
      CLOSE(11)
      END

      SUBROUTINE CAPTURE
      IMPLICIT NONE
""" + common + """      COUNT=COUNT+1
      IDS(COUNT)=NINT(SECPAR(0))
      FIELDS(1,COUNT)=SECPAR(1)*PAMA(IDS(COUNT))
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
        IF(NUSED.EQ.0.AND.FIRST.GE.0.D0) RANDOMS(J)=FIRST
        NUSED=NUSED+1
      ENDDO
      END
"""
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "meson_oracle.f").write_text(driver + rho + angle)
    (args.output / "meson_oracle_provenance.json").write_text(json.dumps({
        "source": str(args.source.resolve()), "sha256": digest,
        "routines": ["RHO0DC pre-cut two-body decay", "ADDANG3"],
        "not_linked_to_cpp": True,
        "changes": ["shower metadata, angular cuts/deposits and stack operations replaced by pre-cut capture",
                    "source kinematics, origin-dependent dipole rejection, ordering and muon polarization retained",
                    "Park-Miller test stream 1; optional first-uniform override exercises rare muon decay",
                    "mass and direction inputs supplied; this is not a full-shower oracle"],
        "source_comment_discrepancy": "KFROM=1 dipole rejection also applies to the muon branch in executable source, despite the routine header's isotropic-muon comment",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
