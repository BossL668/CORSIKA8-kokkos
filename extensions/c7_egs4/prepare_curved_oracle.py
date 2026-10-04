"""Extract original CURVED normal-step frame/time arithmetic as a test oracle.

Independent executable only. It never supplies runtime transport to C++.
Inputs contain an already advanced local endpoint, pre-scatter direction and
the previous observer endpoint. HOWFAR, scattering and shower IO are not here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def routine(source, name):
    begin = source.index("      SUBROUTINE " + name)
    start = source.index("#if __CURVED__\n      IF ( IPASC .EQ. 0  .OR.  .NOT.FFLATOUT ) THEN", begin)
    end = source.index("      IF ( ABS(", source.index("      TIM(NP) = TIM(NP) + TDIFF * SPEED/SPEED0", start))
    block = source[start + len("#if __CURVED__\n"):end]
    # Keep the original local-frame branch; only disable debug output.
    return subprocess.run(["cpp", "-P", "-traditional-cpp", "-D__UPWARD__=1"],
                          input=block, text=True, capture_output=True, check=True).stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    source = a.source.read_text()
    digest = hashlib.sha256(a.source.read_bytes()).hexdigest()
    if digest != "0d52382ad2d604f75ca28195ad0e0f581dde3cc0386ee7ef69bb9045a18bf2fd":
        raise RuntimeError("C7 source changed; re-audit curved geometry")
    common = """      IMPLICIT NONE
      DOUBLE PRECISION Q(22),OUT(13),X(1),Y(1),Z(1),U(1),V(1),W(1),
     * E(1),TIM(1),WA(1),WAP(1),ZAP(1),XXXX(1),YYYY(1),C(50),
     * OBSLEV(1),OBSLVL(1),XOLD,YOLD,XXXOLD,YYYOLD,ZAPOLD,
     * TRANS2,AUXIL,ZNEW,SINDIF,COSDIF,CORR,COSTHENEW,AUXILSQ,
     * AUX2SQ,TANPHI,PHI1,RRR,XXX,YYY,TDIFF,SPEED0,SPEED,
     * VSTEP,USTEP,TVSTEP,TVSTPC,VCI,PRM,WCUT
      INTEGER NP,IPASC,IDISC,MDEBUG,RESULT
      LOGICAL FEGSDB,FFLATOUT,IRETC,FNPRIM
      NP=1
      FEGSDB=.FALSE.
      FNPRIM=.FALSE.
      FFLATOUT=Q(3).NE.0.D0
      IPASC=NINT(Q(2))
      XOLD=Q(4)
      YOLD=Q(5)
      XXXOLD=Q(7)
      YYYOLD=Q(8)
      ZAPOLD=Q(9)
      X(1)=Q(10)
      Y(1)=Q(11)
      Z(1)=Q(12)
      U(1)=Q(13)
      V(1)=Q(14)
      W(1)=Q(15)
      C(1)=Q(16)
      OBSLEV(1)=Q(17)
      OBSLVL(1)=Q(17)
      WCUT=Q(18)
      VSTEP=Q(19)
      USTEP=Q(19)
      TVSTEP=Q(20)
      TVSTPC=Q(20)
      E(1)=Q(21)
      PRM=Q(22)
      VCI=1.D0/2.99792458D10
      TIM=0.D0
      WA=1.D0
      WAP=0.D0
      ZAP=0.D0
      XXXX=0.D0
      YYYY=0.D0
      RESULT=1
"""
    finish = """      OUT=(/X(1),Y(1),Z(1),U(1),V(1),W(1),
     * XXXX(1),YYYY(1),ZAP(1),WA(1),WAP(1),TIM(1),DBLE(RESULT)/)
      END
"""
    electron = routine(source, "ELECTR(")
    # Original labels leave the normal geometry block. Retain their separate
    # dispositions instead of executing the omitted shower endpoint handlers.
    electron = electron.replace("GOTO 420", "GOTO 991").replace("GOTO 498", "GOTO 992")
    electron += """      GOTO 993
 991  RESULT=2
      GOTO 993
 992  RESULT=3
 993  CONTINUE
 560  FORMAT(4E26.17)
 562  FORMAT(2E26.17)
 557  FORMAT(6E26.17)
"""
    photon = routine(source, "PHOTON(").replace("GOTO 1000", "GOTO 991")
    photon += """      GOTO 993
 991  RESULT=2
 993  CONTINUE
"""
    program = """      PROGRAM REFERENCE
      IMPLICIT NONE
      DOUBLE PRECISION Q(22),OUT(13)
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
        WRITE(11,'(13(ES26.17E3,1X))') OUT
      ENDDO
      END
""" + "      SUBROUTINE ELECTRON_REFERENCE(Q,OUT)\n" + common + electron + finish
    program += "      SUBROUTINE PHOTON_REFERENCE(Q,OUT)\n" + common + photon + finish
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "curved_oracle.f").write_text(program)
    (a.output / "curved_oracle_provenance.json").write_text(json.dumps({
        "source": str(a.source.resolve()), "sha256": digest,
        "not_linked_to_cpp": True, "routines": ["ELECTR CURVED normal-step", "PHOTON CURVED normal-step"],
        "scope": "Rebase and time correction only; supplied old observer endpoint, advanced local endpoint and time path"
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
