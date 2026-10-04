"""External C7 ANNIH/PHOTO/ELECTR-cut reference; never a runtime dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    source = args.source.read_text()
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest != "0d52382ad2d604f75ca28195ad0e0f581dde3cc0386ee7ef69bb9045a18bf2fd":
        raise RuntimeError("C7 source changed; re-audit extraction")
    common = """      INTEGER NP,IQ(2),IBLOBE,LELEC,IDR,IRCODE
      DOUBLE PRECISION E(2),U(2),V(2),W(2),PRM,RMI,PEIE,
     * RD(4),TWOPI,SINTHE,COSTHE,SINPHI,COSPHI,THETA,EDEP,FLIP
      REAL EBINDA,AE
      COMMON /C7STATE/ E,U,V,W,PRM,RMI,PEIE,RD,TWOPI,SINTHE,
     * COSTHE,SINPHI,COSPHI,THETA,EDEP,FLIP,EBINDA,AE,
     * NP,IQ,IBLOBE,LELEC,IDR,IRCODE
      INTEGER*8 RANDOM_STATE
      INTEGER NUSED
      COMMON /RANDOMSTATE/ RANDOM_STATE,NUSED
"""
    routines = []
    for name in ("ANNIH", "PHOTO", "UPHI", "STOPPED"):
        if name == "STOPPED":
            a = source.index(" 390  IF ( PEIE .GT. AE ) THEN")
            b = source.index("C  ELECTRON IS ELEMINATED BECAUSE OF CUT", a)
            part = "      SUBROUTINE STOPPED\n      IMPLICIT NONE\n" + common + source[a:b] + "      END\n"
            # Metadata is outside this local particle/energy fixture.
            a = part.index("          X(NP)  = X(NP-1)")
            b = part.index("C  SECOND GAMMA IN OPPOSITE DIRECTION", a)
            part = part[:a] + part[b:]
        else:
            a = source.index("      SUBROUTINE " + name + ("(" if name == "UPHI" else "\n"))
            b = source.index("\n      END\n", a) + len("\n      END\n")
            part = source[a:b]
            part = re.sub(r'#define .*?\n#include "corsika.h"\n', common, part, count=1, flags=re.S)
            part = re.sub(r"      IF \( FEGSDB \) THEN\n.*?      ENDIF\n", "", part, count=1, flags=re.S)
            part = part.replace("      IGEN(NP) = IGEN(NP) + 1000000\n", "")
            if name == "UPHI":
                a = part.index("      X(NP) = X(NP-1)")
                b = part.index(" 1130 CONTINUE", a)
                part = part[:a] + part[b:]
        part = subprocess.run(["cpp", "-P", "-traditional-cpp"], input=part,
                              text=True, capture_output=True, check=True).stdout
        part = re.sub(r"      IF \( LLONGI \) THEN\n.*?      ENDIF\n", "", part, flags=re.S)
        if "#" in part or re.search(r"\b(FEGSDB|IGEN|LLONGI|WTM|ZAP)\b", part):
            raise RuntimeError("Unexpected stateful code in " + name)
        routines.append(part)
    driver = """      PROGRAM REFERENCE
      IMPLICIT NONE
""" + common + """      INTEGER N,I,PROCESS,PID,J,COUNT
      CHARACTER*1024 INPUT,OUTPUT
      CALL GET_COMMAND_ARGUMENT(1,INPUT)
      CALL GET_COMMAND_ARGUMENT(2,OUTPUT)
      OPEN(10,FILE=TRIM(INPUT),STATUS='OLD')
      OPEN(11,FILE=TRIM(OUTPUT),STATUS='REPLACE')
      READ(10,*) N
      DO I=1,N
        E=0.D0
        U=0.D0
        V=0.D0
        W=0.D0
        IQ=1
        READ(10,*) PROCESS,PID,E(1),PRM,EBINDA,
     *    U(1),V(1),W(1),RANDOM_STATE
        NP=1
        NUSED=0
        EDEP=0.D0
        PEIE=E(1)
        RMI=1.D0/PRM
        TWOPI=2.D0*ACOS(-1.D0)
        AE=REAL(PRM+.4D0)
        IF(PROCESS.EQ.0) THEN
          IQ(1)=2
          CALL ANNIH
          COUNT=NP
        ELSEIF(PROCESS.EQ.1) THEN
          IQ(1)=1
          CALL PHOTO
          COUNT=1-IBLOBE
        ELSEIF(PROCESS.EQ.2) THEN
          LELEC=-1
          IQ(1)=3
          IF(PID.EQ.-11) THEN
            LELEC=1
            IQ(1)=2
          ENDIF
          CALL STOPPED
          COUNT=NP
        ELSE
          STOP 1
        ENDIF
        WRITE(11,'(4(I8,1X),9(ES26.17E3,1X))')
     *    COUNT,NUSED,IQ(1),IQ(2),EDEP,
     *    E(1),U(1),V(1),W(1),E(2),U(2),V(2),W(2)
      ENDDO
      END

      SUBROUTINE RMMARD(RANDOMS,N,ISEQ)
      IMPLICIT NONE
      INTEGER N,ISEQ,I,NUSED
      INTEGER*8 RANDOM_STATE
      DOUBLE PRECISION RANDOMS(N)
      COMMON /RANDOMSTATE/ RANDOM_STATE,NUSED
      DO I=1,N
        IF(NUSED.GE.100000) STOP 3
        RANDOM_STATE=MOD(RANDOM_STATE*48271_8,2147483647_8)
        RANDOMS(I)=DBLE(RANDOM_STATE)/2147483647.D0
        NUSED=NUSED+1
      ENDDO
      END
"""
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "termination_oracle.f").write_text(driver + "\n".join(routines))
    (args.output / "termination_oracle_provenance.json").write_text(json.dumps({
        "source": str(args.source.resolve()), "sha256": digest,
        "routines": ["ANNIH", "PHOTO", "UPHI", "ELECTR label 390"],
        "not_linked_to_cpp": True,
        "changes": ["separate executable with supplied random stream",
                    "metadata and optional histogram accumulation omitted; EDEP retained",
                    "two-slot output fixture, no propagation"],
        "source_caveat": "ANNIH uses same azimuth and positive sine for both photons; preserved literally",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
