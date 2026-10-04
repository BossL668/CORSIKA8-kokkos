"""Extract original C7 detector-approach arithmetic for an independent test.

No C7 executable or Fortran library is called by the C++ backend.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def block(source, name):
    begin = source.index("      SUBROUTINE " + name)
    start = source.index("      DISTO2 = X(NP)**2 + Y(NP)**2", begin)
    end = source.index("#else\n#if __SLANT__", start)
    text = source[start:end]
    return subprocess.run(["cpp", "-P", "-traditional-cpp", "-D__UPWARD__=1"],
                          input=text, text=True, capture_output=True, check=True).stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    digest = hashlib.sha256(a.source.read_bytes()).hexdigest()
    if digest != "0d52382ad2d604f75ca28195ad0e0f581dde3cc0386ee7ef69bb9045a18bf2fd":
        raise RuntimeError("C7 source changed; re-audit observation preparation")
    source = a.source.read_text()
    common = """      IMPLICIT NONE
      DOUBLE PRECISION Q(16),OUT(12),X(1),Y(1),Z(1),U(1),V(1),W(1),
     * WA(1),WAP(1),ZAP(1),C(50),OBSLEV(1),OBSLVL(1),WCUT,
     * DISTO2,AUXILSQ,PHI,PHIC,DSTEFF,SINDIF,COSDIF,COSTHENEW,
     * PHI1,SINTEA,RRR,TANPHI,USTEP
      INTEGER NP,IPASC,IDISC,MDEBUG,RESULT
      LOGICAL FEGSDB,FFLATOUT,IRETC
      NP=1
      FEGSDB=.FALSE.
      FFLATOUT=Q(2).NE.0.D0
      X(1)=Q(3)
      Y(1)=Q(4)
      Z(1)=Q(5)
      U(1)=Q(6)
      V(1)=Q(7)
      W(1)=Q(8)
      WA(1)=Q(9)
      WAP(1)=Q(10)
      ZAP(1)=Q(11)
      C(1)=Q(12)
      C(3)=Q(13)
      OBSLEV(1)=Q(14)
      OBSLVL(1)=Q(14)
      WCUT=Q(15)
      USTEP=Q(16)
      IDISC=-1
      IPASC=0
      RESULT=1
      IRETC=.TRUE.
"""
    finish = """      GOTO 992
 991  RESULT=2
 992  CONTINUE
      OUT=(/X(1),Y(1),Z(1),U(1),V(1),W(1),ZAP(1),
     * WA(1),WAP(1),USTEP,DBLE(IPASC),DBLE(RESULT)/)
      END
"""
    program = """      PROGRAM REFERENCE
      IMPLICIT NONE
      DOUBLE PRECISION Q(16),OUT(12)
      INTEGER I,N
      CHARACTER*1024 INPUT,OUTPUT
      CALL GET_COMMAND_ARGUMENT(1,INPUT)
      CALL GET_COMMAND_ARGUMENT(2,OUTPUT)
      OPEN(10,FILE=TRIM(INPUT),STATUS='OLD')
      OPEN(11,FILE=TRIM(OUTPUT),STATUS='REPLACE')
      READ(10,*) N
      DO I=1,N
        READ(10,*) Q
        IF(Q(1).EQ.0.D0)THEN
          CALL ELECTRON_REFERENCE(Q,OUT)
        ELSE
          CALL PHOTON_REFERENCE(Q,OUT)
        ENDIF
        WRITE(11,'(12(ES26.17E3,1X))') OUT
      ENDDO
      END
      SUBROUTINE AUSGB2
      END
"""
    for name, sub in [("ELECTR(", "ELECTRON_REFERENCE"), ("PHOTON(", "PHOTON_REFERENCE")]:
        code = block(source, name).replace("GOTO 420", "GOTO 991").replace("GOTO 1000", "GOTO 991")
        program += "      SUBROUTINE " + sub + "(Q,OUT)\n" + common + code + finish
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "observation_oracle.f").write_text(program)
    (a.output / "observation_oracle_provenance.json").write_text(json.dumps({
        "source": str(a.source.resolve()), "sha256": digest, "not_linked_to_cpp": True,
        "scope": "ELECTR/PHOTON IDISC=-1 pre-transport detector coordinate and remaining-step conversion"
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
