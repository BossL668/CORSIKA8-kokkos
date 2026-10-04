"""Original C7 omega/phi RESDEC + DECAY6, independent pre-cut reference."""
from pathlib import Path
import argparse
import hashlib
import json
import re
from prepare_two_body_oracle import extract_two_body_source


def preprocess(text):
    """Only fixture flags: UPWARD on, other optional output/neutrino flags off."""
    stack = []
    active = True
    result = []
    for line in text.splitlines(True):
        if line.startswith("#if "):
            condition = line[4:].strip()
            invert = condition.startswith("!")
            flag = condition[1:] if invert else condition
            if not re.fullmatch(r"__[A-Z0-9]+__", flag):
                raise RuntimeError("Unexpected conditional " + condition)
            value = flag == "__UPWARD__"
            if invert:
                value = not value
            stack.append((active, value))
            active = active and value
        elif line.startswith("#else"):
            parent, value = stack[-1]
            active = parent and not value
        elif line.startswith("#endif"):
            active = stack.pop()[0]
        elif line.startswith("#"):
            raise RuntimeError("Unexpected preprocessor line " + line)
        elif active:
            result.append(line)
    if stack:
        raise RuntimeError("Unclosed source preprocessor block")
    return "".join(result)


def pre_cut_angles(text):
    # Group fixed-form continuation lines, so nested multiline IFs in
    # deposition blocks are counted correctly. Remove only angular gates
    # and their rejected/deposition arms, preserving accepted source code.
    statements = []
    for line in text.splitlines(True):
        if line[:1] in ("C", "c", "*", "!") or not line.strip():
            continue
        if len(line) > 5 and line[5] not in (" ", "0"):
            statements[-1][0] += " " + line[6:].strip()
            statements[-1][1] += line
        else:
            statements.append([line[6:].strip(), line])
    result = []
    depth = 0
    accepting = True
    removed = 0
    for statement, raw in statements:
        statement = statement.split("!", 1)[0].strip()
        opening = bool(re.match(r"IF\s*\(.*\)\s*THEN$", statement))
        closing = statement == "ENDIF"
        if not depth:
            if opening and "SECPAR(2) .GE. C(29)" in statement:
                depth = 1
                accepting = True
                removed += 1
            else:
                result.append(raw)
            continue
        if closing:
            depth -= 1
            if not depth:
                continue
        elif opening:
            depth += 1
        elif depth == 1 and statement == "ELSE":
            accepting = False
            continue
        if accepting:
            result.append(raw)
    if depth or removed != 4:
        raise RuntimeError(f"RESDEC angular extraction changed: depth={depth}, gates={removed}")
    return "".join(result)


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
     * POLART,POLARF,FIELDS(6,3),GAMMA,BETA,COSTHE,PHIX,PHIY,
     * GAM345(3),COS345(3),PHI345(3),FIRST
      INTEGER RS,NUSED,COUNT,IDS(3)
      COMMON /DSTATE/ SECPAR,PAMA,RD,PI,PI2,POLART,POLARF,
     * FIELDS,GAMMA,BETA,COSTHE,PHIX,PHIY,GAM345,COS345,
     * PHI345,FIRST
      COMMON /ISTATE/ RS,NUSED,COUNT,IDS
"""
    start = source.index("      ELSEIF ( ITYPE .EQ. 50 ) THEN", source.index("      SUBROUTINE RESDEC"))
    end = source.index("C  EXCITED KAON RESONANCES", start)
    selected = pre_cut_angles(preprocess(source[start:end]))
    selected = selected.replace("ELSEIF ( ITYPE .EQ. 50 )", "IF ( ITYPE .EQ. 50 )", 1)
    selected = selected.replace("CALL TSTACK", "CALL CAPTURE")
    resdec = "      SUBROUTINE RESDEC(ITYPE)\n      IMPLICIT NONE\n" + common + """      INTEGER ITYPE,I
""" + selected + "      ENDIF\n      RETURN\n      END\n"
    start = source.index("      SUBROUTINE DECAY6(")
    end = source.index("\n      END\n", start) + len("\n      END\n")
    three = re.sub(r'#define .*?\n#include "corsika.h"\n', common + """      LOGICAL DEBUG
      PARAMETER (DEBUG=.FALSE.)
      INTEGER MONIOU,MDEBUG
      PARAMETER (MONIOU=6,MDEBUG=6)
      DOUBLE PRECISION OB3
      PARAMETER (OB3=1.D0/3.D0)
""", source[start:end], count=1, flags=re.S)
    three = preprocess(three)
    decays = extract_two_body_source(source, common)
    driver = """      PROGRAM REFERENCE
      IMPLICIT NONE
""" + common + """      INTEGER N,I,J,K,ITYPE
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
        GAM345=0.D0
        COS345=0.D0
        PHI345=0.D0
        IDS=0
        READ(10,*) ITYPE,TOTAL,PAMA(50),PAMA(49),PAMA(8),
     *   PAMA(7),PAMA(5),PAMA(11),PAMA(10),PAMA(16),PAMA(17),
     *   PHIX,PHIY,COSTHE,RS,FIRST
        GAMMA=TOTAL/PAMA(ITYPE)
        PAMA=PAMA*0.001D0
        PAMA(9)=PAMA(8)
        PAMA(6)=PAMA(5)
        PAMA(12)=PAMA(11)
        PHIY=-PHIY
        BETA=SQRT((GAMMA-1.D0)*(GAMMA+1.D0))/GAMMA
        PI=ACOS(-1.D0)
        PI2=2.D0*PI
        NUSED=0
        COUNT=0
        CALL RESDEC(ITYPE)
        WRITE(11,*) ITYPE,COUNT,NUSED,RS,IDS,
     *    ((FIELDS(K,J),K=1,6),J=1,3)
      ENDDO
      CLOSE(10)
      CLOSE(11)
      END

      SUBROUTINE CAPTURE
      IMPLICIT NONE
""" + common + """      COUNT=COUNT+1
      IF(COUNT.GT.3) STOP 32
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
        IF(NUSED.EQ.0.AND.FIRST.GE.0.D0) RANDOMS(J)=FIRST
        NUSED=NUSED+1
      ENDDO
      END
"""
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "resonance_oracle.f").write_text(driver + resdec + three + decays)
    (args.output / "resonance_oracle_provenance.json").write_text(json.dumps({
        "source": str(args.source.resolve()), "sha256": digest,
        "routines": ["RESDEC omega/phi", "DECAY6", "DECAY1", "DECAY2", "ADDANG3"],
        "not_linked_to_cpp": True,
        "changes": ["Source RESDEC thresholds/calls/order retained, four angular gates and rejected deposit arms removed",
                    "TSTACK replaced with pre-cut capture; no thinning, shower metadata or host transport",
                    "DECAY6 arithmetic retained (omega/phi call only uniform MODE=2); optional output flags disabled",
                    "Same extracted two-body routines as independent DECAY1/2 tests",
                    "Park-Miller stream 1 plus optional first draw override; total MeV -> C7 GeV mass convention"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
