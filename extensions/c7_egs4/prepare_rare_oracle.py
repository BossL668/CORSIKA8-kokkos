"""Standalone C7 rare-vertex reference; NEVER linked to native C++ runtime.

Preserve ELNUCL/MUPAIR/UPHI arithmetic mechanically from the checked source.
Replace NKG, metadata and host stack/generator calls by observation stubs;
PIGEN is not evaluated by this fixture. Random streams use supplied test seeds.
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
    if digest != "0d52382ad2d604f75ca28195ad0e0f581dde3cc0386ee7ef69bb9045a18bf2fd":
        raise RuntimeError("Changed original C7 source; re-audit extraction")
    common = """      DOUBLE PRECISION E(2),U(2),V(2),W(2),RD(4),PRM,PRRMMU,
     * COMPOS(3),PAMA(14),ELCUT(4),AVERAW,PI,TWOPI,SE,PITHR,
     * SINTHE,COSTHE,COSPHI,SINPHI,THETA,POLART,POLARF,
     * OUTPOLT(2),OUTPOLF(2)
      COMMON /DSTATE/ E,U,V,W,RD,PRM,PRRMMU,COMPOS,PAMA,ELCUT,
     * AVERAW,PI,TWOPI,SE,PITHR,SINTHE,COSTHE,COSPHI,SINPHI,
     * THETA,POLART,POLARF,OUTPOLT,OUTPOLF
      INTEGER NP,IQ(2),NUSED,RS,TARGETOUT,NREQUEST
      COMMON /ISTATE/ NP,IQ,NUSED,RS,TARGETOUT,NREQUEST
"""
    routines = []
    for name in ("MUPAIR", "ELNUCL", "UPHI"):
        start = source.index("      SUBROUTINE " + name + ("(" if name == "UPHI" else "\n"))
        end = source.index("\n      END\n", start) + len("\n      END\n")
        part = source[start:end]
        part = re.sub(r'#define .*?\n#include "corsika.h"\n', common, part, count=1, flags=re.S)
        part = re.sub(r"      IF \(\s*FEGSDB\s*\) THEN\n.*?      ENDIF\n", "", part, count=1, flags=re.S)
        part = re.sub(r"      IF \( DEBUG \).*\n", "", part)
        if name == "MUPAIR":
            a = part.index("C  INCREMENT GENERATION COUNTER")
            b = part.index("C  SELECT TARGET NUCLEUS", a)
            part = part[:a] + "      PEIG = E(NP)\n" + part[b:]
            part = part.replace("C  BOUNDARIES OF INTEGRATION", "      TARGETOUT = JE\nC  BOUNDARIES OF INTEGRATION")
            part = part.replace("      INT_ICOUNT = 0\n", "").replace("      CALL TSTEND\n", "")
        if name == "UPHI":
            part = part.replace("#if __MULTITHIN__\n      INTEGER          IK\n#endif\n", "")
            a = part.index("      X(NP) = X(NP-1)")
            b = part.index(" 1130 CONTINUE", a)
            part = part[:a] + part[b:]
        if "#" in part or "CALL NKG" in part or "IF ( FEGSDB" in part:
            raise RuntimeError(f"Unremoved source scaffolding: {name}")
        routines.append(part)
    driver = """      PROGRAM REFERENCE
      IMPLICIT NONE
""" + common + """      INTEGER N,I,J,IP,PDG,COUNT,RSIN
      DOUBLE PRECISION ENERGY,CUT
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
        IQ=0
        OUTPOLT=0.D0
        OUTPOLF=0.D0
        READ(10,*) IP,PDG,ENERGY,PRM,PRRMMU,PAMA(8),PAMA(14),
     *   AVERAW,COMPOS,CUT,PITHR,U(1),V(1),W(1),RSIN
        E(1)=ENERGY
        PAMA(2)=PRM*1.D-3
        PAMA(8)=PAMA(8)*1.D-3
        PAMA(14)=PAMA(14)*1.D-3
        ELCUT(3)=CUT*1.D-3
        NP=1
        NUSED=0
        RS=RSIN
        TARGETOUT=0
        NREQUEST=0
        PI=ACOS(-1.D0)
        TWOPI=2.D0*PI
        SE=SQRT(EXP(1.D0))
        IF(IP.EQ.0) THEN
          IQ(1)=1
          CALL MUPAIR
          COUNT=2
          DO J=1,2
            IF(IQ(J).EQ.5) THEN
              IQ(J)=-13
            ELSE
              IQ(J)=13
            ENDIF
          ENDDO
        ELSE
          IQ(1)=3
          IF(PDG.EQ.-11) IQ(1)=2
          CALL ELNUCL
          COUNT=NP
          IQ(1)=PDG
          IQ(2)=22
        ENDIF
        WRITE(11,'(7(I12,1X),12(ES26.17E3,1X))')
     *   IP,COUNT,NUSED,RS,TARGETOUT,IQ(1),IQ(2),
     *   E(1),U(1),V(1),W(1),E(2),U(2),V(2),W(2),
     *   OUTPOLT(1),OUTPOLF(1),OUTPOLT(2),OUTPOLF(2)
      ENDDO
      CLOSE(10)
      CLOSE(11)
      END

      SUBROUTINE MUPROP
      IMPLICIT NONE
""" + common + """      OUTPOLT(NP)=POLART
      OUTPOLF(NP)=POLARF
      NREQUEST=NREQUEST+1
      NP=NP-1
      END

      SUBROUTINE PIGEN(REALGAMMA)
      IMPLICIT NONE
      LOGICAL REALGAMMA
""" + common + """      NREQUEST=NREQUEST+1
C     Capture the virtual photon before hadronic generation.
      END

      SUBROUTINE RMMARD(RANDOMS,N,ISEQ)
      IMPLICIT NONE
      INTEGER N,ISEQ,I
      DOUBLE PRECISION RANDOMS(N)
""" + common + """      DO I=1,N
        RS=INT(MOD(16807_8*INT(RS,8),2147483647_8))
        RANDOMS(I)=DBLE(RS)/2147483647.D0
      ENDDO
      NUSED=NUSED+N
      END
"""
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rare_oracle.f").write_text(driver + "\n".join(routines))
    (args.output / "rare_oracle_provenance.json").write_text(json.dumps({
        "source": str(args.source.resolve()), "sha256": digest,
        "routines": ["MUPAIR", "ELNUCL", "UPHI"], "not_linked_to_cpp": True,
        "changes": ["Metadata/NKG/debug omitted; two-slot stack",
                    "MUPROP captures generated muons/polarization before cuts/host transport",
                    "PIGEN captures virtual photon, does not model hadronic generation",
                    "External deterministic uniform provider; not original C7 shower RNG streams"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
