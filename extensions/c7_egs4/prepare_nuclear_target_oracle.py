"""Extract original SDPM photon target selection, not a hadron generator."""
import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != "0d52382ad2d604f75ca28195ad0e0f581dde3cc0386ee7ef69bb9045a18bf2fd":
        raise RuntimeError("C7 source changed; re-audit extraction")
    start = source.index("      SUBROUTINE SDPM( LTA )")
    start = source.index("          FRACTN = COMPOS(1) * 11.04019D0", start)
    end = source.index("C  TARGET IS PREDETERMINED", start)
    body = source[start:end]
    if "#" in body:
        raise RuntimeError("Unexpected conditional target implementation")
    driver = """      PROGRAM REFERENCE
      IMPLICIT NONE
      DOUBLE PRECISION COMPOS(3),FRACTN,FRCTNO,SIGAIR,TAR,RD(1),U
      INTEGER N,I,LIT,NUSED
      CHARACTER*1024 INPUT,OUTPUT
      COMMON /RANDOMINPUT/ U,NUSED
      CALL GET_COMMAND_ARGUMENT(1,INPUT)
      CALL GET_COMMAND_ARGUMENT(2,OUTPUT)
      OPEN(10,FILE=TRIM(INPUT),STATUS='OLD')
      OPEN(11,FILE=TRIM(OUTPUT),STATUS='REPLACE')
      READ(10,*) N
      DO I=1,N
        READ(10,*) COMPOS,U
        NUSED=0
""" + body + """        WRITE(11,*) NINT(TAR),NUSED
      ENDDO
      CLOSE(10)
      CLOSE(11)
      END
      SUBROUTINE RMMARD(RANDOMS,N,ISEQ)
      IMPLICIT NONE
      INTEGER N,ISEQ,NUSED
      DOUBLE PRECISION RANDOMS(N),U
      COMMON /RANDOMINPUT/ U,NUSED
      IF(N.NE.1.OR.ISEQ.NE.1.OR.NUSED.NE.0) STOP 31
      RANDOMS(1)=U
      NUSED=1
      END
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(driver)


if __name__ == "__main__":
    main()
