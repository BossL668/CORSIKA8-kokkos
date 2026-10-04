"""Extract a SEPARATE pre-cut PIGEN1/PIGEN2/RHOGEN/PTRANS/UPHI reference.

Runtime uses only native C++. This fixture retains the original arithmetic,
captures particles before decay/cuts/stack handling, and supplies separate
test RNG streams. It is not a full C7 photohadronic transport oracle.
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
        raise RuntimeError("Changed C7 source; audit extraction again")
    common = """      DOUBLE PRECISION E(3),U(3),V(3),W(3),RD(4),PAMA(51),
     * AMASPR,AMASNT,TWOPI,SINTHE,COSTHE,SINPHI,COSPHI,THETA
      COMMON /DSTATE/ E,U,V,W,RD,PAMA,AMASPR,AMASNT,TWOPI,
     * SINTHE,COSTHE,SINPHI,COSPHI,THETA
      INTEGER NP,IQ(3),NUSED(2),RS(2),TARGET,BRANCH
      COMMON /ISTATE/ NP,IQ,NUSED,RS,TARGET,BRANCH
"""
    routines = []
    for name in ("PIGEN1", "PIGEN2", "RHOGEN", "PTRANS", "UPHI", "PIGEN"):
        head = "      DOUBLE PRECISION FUNCTION PTRANS()" if name == "PTRANS" else (
            "      SUBROUTINE " + name + ("(" if name in ("UPHI", "PIGEN") else "\n"))
        start = source.index(head)
        end = source.index("\n      END\n", start) + len("\n      END\n")
        part = source[start:end]
        part = re.sub(r'#define .*?\n#include "corsika.h"\n', common, part, count=1, flags=re.S)
        if name in ("PIGEN1", "PIGEN2", "RHOGEN"):
            # Keep declarations, then original kinematics through first UPHI;
            # append the original recoil calculation without its host-cut IF.
            declarations = part[:part.index("#if __AUGERHIST__")]
            a = part.index("      PEIG = E(NP)")
            marker = "      IF ( E(NP)-AMASS5 .GT." if name == "PIGEN2" else "      IF ( ENUCL-AMASS4 .GT."
            b = part.index(marker, a)
            body = part[a:b]
            # RHOGEN diagnostic WRITE has conditional first line and common
            # continuation lines. Remove whole diagnostic statement first.
            body = re.sub(r"#if __INTTEST__\n      IF \([^\n]*\n#else\n[^\n]*\n#endif\n(?:     \*[^\n]*\n)+", "", body)
            body = re.sub(r"#if __INTTEST__\n.*?#endif\n", "", body, flags=re.S)
            recoil = ""
            if name != "PIGEN2":
                a = part.index("        NP = NP+1" if name == "PIGEN1" else "        AMOM4  =", b)
                b = part.index("        CALL UPHI( 3,2 )", a) + len("        CALL UPHI( 3,2 )\n")
                recoil = part[a:b]
            part = declarations + "      EXTERNAL PTRANS\n" + body + recoil + """      IF(AMASS2.EQ.AMASPR) THEN
        TARGET=2212
      ELSE
        TARGET=2112
      ENDIF
      RETURN
      END
"""
        elif name == "PIGEN":
            a = part.index("      CALL RMMARD( RD,1,2 )")
            b = part.index("C  RESTORE CURPAR PARTICLE", a)
            body = part[a:b]
            a = body.index("C  AT HIGHER ENERGIES")
            b = body.index("C  END OF MANY PION GENERATION", a)
            body = body[:a] + "          BRANCH=4\n" + body[b:]
            body = re.sub(r"#if __INTTEST__\n.*?#endif\n", "", body, flags=re.S)
            body = re.sub(r"        IF \( FEGSDB .*\n     \*.*\n", "", body)
            body = re.sub(r" *INT_ICOUNT = 0\n| *CALL TSTEND\n", "", body)
            for branch, routine in enumerate(("PIGEN1", "PIGEN2", "RHOGEN"), 1):
                body = body.replace("CALL " + routine, "BRANCH=" + str(branch))
            part = "      SUBROUTINE SELECTPIGEN\n      IMPLICIT NONE\n" + common + """      DOUBLE PRECISION PEIG,AUXIL,ECMVM,VMFRAC
      PEIG=E(1)
""" + body + "      RETURN\n      END\n"
        elif name == "PTRANS":
            # Original C block-data defaults, held fixed for this executable.
            part = part.replace("C(34)", "20.D0").replace("C(12)", "0.1D0")
            part = re.sub(r"[ C]     IF \( DEBUG \).*\n", "", part)
        else:
            part = part.replace("#if __MULTITHIN__\n      INTEGER          IK\n#endif\n", "")
            a = part.index("      X(NP) = X(NP-1)")
            b = part.index(" 1130 CONTINUE", a)
            part = part[:a] + part[b:]
        if re.search(r"^\s*#", part, flags=re.M) or re.search(r"\b(FEGSDB|MDEBUG|SECPAR|ELCUT)\b", part):
            raise RuntimeError(f"Unremoved scaffolding in {name}")
        routines.append(part)
    driver = """      PROGRAM REFERENCE
      IMPLICIT NONE
""" + common + """      INTEGER N,I,J,PROCESS,COUNT
      DOUBLE PRECISION PT,PTRANS
      CHARACTER*1024 INPUT,OUTPUT
      EXTERNAL PTRANS
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
        READ(10,*) PROCESS,E(1),PAMA(14),PAMA(13),PAMA(7),
     *    PAMA(8),PAMA(49),PAMA(50),PAMA(51),
     *    U(1),V(1),W(1),RS(1),RS(2)
        PAMA(9)=PAMA(8)
        AMASPR=PAMA(14)*1.D3
        AMASNT=PAMA(13)*1.D3
        TWOPI=2.D0*ACOS(-1.D0)
        NP=1
        NUSED=0
        TARGET=0
        BRANCH=0
        COUNT=0
        IF(PROCESS.EQ.0) THEN
          PT=PTRANS()
          E=0.D0
          U=0.D0
          V=0.D0
          W=0.D0
          E(1)=PT
        ELSEIF(PROCESS.EQ.1) THEN
          BRANCH=1
          CALL PIGEN1
          COUNT=2
        ELSEIF(PROCESS.EQ.2) THEN
          BRANCH=2
          CALL PIGEN2
          COUNT=3
        ELSEIF(PROCESS.EQ.3) THEN
          BRANCH=3
          CALL RHOGEN
          COUNT=2
        ELSE
          CALL SELECTPIGEN
          IF(PROCESS.EQ.5) THEN
            IF(BRANCH.EQ.1) CALL PIGEN1
            IF(BRANCH.EQ.2) CALL PIGEN2
            IF(BRANCH.EQ.3) CALL RHOGEN
            IF(BRANCH.NE.4) COUNT=NP
          ENDIF
          IF(COUNT.EQ.0) THEN
            E=0.D0
            U=0.D0
            V=0.D0
            W=0.D0
          ENDIF
        ENDIF
        WRITE(11,*) PROCESS,TARGET,BRANCH,COUNT,NUSED,RS,IQ,
     *    (E(J),U(J),V(J),W(J),J=1,3)
      ENDDO
      CLOSE(10)
      CLOSE(11)
      END

      SUBROUTINE RMMARD(RANDOMS,N,ISEQ)
      IMPLICIT NONE
      INTEGER N,ISEQ,J,NP,IQ(3),NUSED(2),RS(2),TARGET,BRANCH
      DOUBLE PRECISION RANDOMS(N)
      COMMON /ISTATE/ NP,IQ,NUSED,RS,TARGET,BRANCH
      DO J=1,N
        RS(ISEQ)=MOD(16807_8*RS(ISEQ),2147483647_8)
        RANDOMS(J)=DBLE(RS(ISEQ))/2147483647.D0
      ENDDO
      NUSED(ISEQ)=NUSED(ISEQ)+N
      END
"""
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "photonuclear_oracle.f").write_text(driver + "\n".join(routines))
    (args.output / "photonuclear_oracle_provenance.json").write_text(json.dumps({
        "source": str(args.source.resolve()), "sha256": digest,
        "routines": ["PIGEN1/PIGEN2/RHOGEN pre-cut kinematics", "PTRANS", "UPHI", "PIGEN branch selection"],
        "not_linked_to_cpp": True,
        "changes": ["pre-cut/pre-decay state retained; recoil-cut guard removed",
                    "PIGEN generator calls replaced by explicit branch identifiers; SDPM is not evaluated",
                    "debug, shower metadata, PIPROP and TSTACK omitted",
                    "PTRANS original C(12)=0.1 GeV, C(34)=20 held fixed",
                    "separate stream-1/stream-2 Park-Miller test providers, not C7 shower RNG"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
