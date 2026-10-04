"""Extract an external C7 reference executable, never linked to the C++ backend.

Only I/O, debug prints, metadata copies and the random provider are replaced.
The sampled energy/angle arithmetic and stack sorting remain from corsika.F.
Generated reference files go to the test build directory, not production.
"""
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
    expected = "0d52382ad2d604f75ca28195ad0e0f581dde3cc0386ee7ef69bb9045a18bf2fd"
    if digest != expected:
        raise RuntimeError("C7 source changed; re-audit extraction before use")
    args.output.mkdir(parents=True, exist_ok=True)
    common = """      INTEGER NP,IQ(2)
      DOUBLE PRECISION E(2),U(2),V(2),W(2),PRM,RMI,RMSQ,
     * TE,THMOLL,RD(4),TWOPI,SINTHE,COSTHE,SINPHI,COSPHI,THETA
      COMMON /C7STATE/ E,U,V,W,PRM,RMI,RMSQ,TE,THMOLL,RD,
     * TWOPI,SINTHE,COSTHE,SINPHI,COSPHI,THETA,NP,IQ
      INTEGER NUSED,NS
      DOUBLE PRECISION TAPE(4096)
      COMMON /RANDOMTAPE/ TAPE,NUSED,NS
"""
    # This is a mechanical fixture extraction, not a hand-transcribed oracle.
    routines = []
    for name in ("COMPT", "MOLLER", "BHABHA", "UPHI"):
        begin = source.index("      SUBROUTINE " + name + ("(" if name == "UPHI" else "\n"))
        end = source.index("\n      END\n", begin) + len("\n      END\n")
        part = source[begin:end]
        part = re.sub(r'#define .*?\n#include "corsika.h"\n', common,
                      part, count=1, flags=re.S)
        part = re.sub(r"      IF \( FEGSDB \) THEN\n.*?      ENDIF\n", "",
                      part, count=1, flags=re.S)
        part = part.replace("      IGEN(NP) = IGEN(NP) + 1000000\n", "")
        part = part.replace("      IGEN(NP) = IGEN(NP) + 1000\n", "")
        if name == "UPHI":
            part = part.replace("#if __MULTITHIN__\n      INTEGER          IK\n#endif\n", "")
            a = part.index("      X(NP) = X(NP-1)")
            b = part.index(" 1130 CONTINUE", a)
            part = part[:a] + part[b:]
        if "#" in part or re.search(r"\b(FEGSDB|IGEN)\b", part):
            raise RuntimeError(f"Unremoved non-kernel code in {name}")
        routines.append(part)
    driver = """      PROGRAM REFERENCE
      IMPLICIT NONE
""" + common + """      INTEGER N,I,J,IP
      CHARACTER*1024 INPUT,OUTPUT
      CALL GET_COMMAND_ARGUMENT(1,INPUT)
      CALL GET_COMMAND_ARGUMENT(2,OUTPUT)
      OPEN(10,FILE=TRIM(INPUT),STATUS='OLD')
      OPEN(11,FILE=TRIM(OUTPUT),STATUS='REPLACE')
      READ(10,*) N
      DO I=1,N
        READ(10,*) IP,E(1),PRM,TE,U(1),V(1),W(1),NS
        IF(NS.GT.4096) STOP 1
        READ(10,*) (TAPE(J),J=1,NS)
        NP=1
        NUSED=0
        RMI=1.D0/PRM
        RMSQ=PRM*PRM
        THMOLL=2.D0*TE+PRM
        TWOPI=2.D0*ACOS(-1.D0)
        IF(IP.EQ.0) THEN
          IQ(1)=1
          CALL COMPT
        ELSEIF(IP.EQ.1) THEN
          IQ(1)=3
          CALL MOLLER
        ELSEIF(IP.EQ.2) THEN
          IQ(1)=2
          CALL BHABHA
        ELSE
          STOP 2
        ENDIF
        WRITE(11,'(4(I8,1X),8(ES26.17E3,1X))')
     *  NP,NUSED,IQ(1),IQ(2),E(1),U(1),V(1),W(1),
     *  E(2),U(2),V(2),W(2)
      ENDDO
      CLOSE(10)
      CLOSE(11)
      END

      SUBROUTINE RMMARD(RANDOMS,N,ISEQ)
      IMPLICIT NONE
      INTEGER N,ISEQ,NUSED,NS
      DOUBLE PRECISION RANDOMS(N),TAPE(4096)
      COMMON /RANDOMTAPE/ TAPE,NUSED,NS
      IF(NUSED+N.GT.NS) STOP 3
      RANDOMS(:)=TAPE(NUSED+1:NUSED+N)
      NUSED=NUSED+N
      END
"""
    (args.output / "collision_oracle.f").write_text(driver + "\n".join(routines))
    (args.output / "collision_oracle_provenance.json").write_text(json.dumps({
        "source": str(args.source.resolve()), "sha256": digest,
        "routines": ["COMPT", "MOLLER", "BHABHA", "UPHI"],
        "not_linked_to_cpp": True,
        "changes": ["random provider replaced by external uniform tape",
                    "debug and metadata copies omitted; two-slot test stack",
                    "standalone executable writes reference results"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
