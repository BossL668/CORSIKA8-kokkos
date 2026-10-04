"""Unlinked original HOWFAR oracle; one observation level, UPWARD, no UPWARDOLD."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument("source",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    source=a.source.read_text();digest=hashlib.sha256(a.source.read_bytes()).hexdigest()
    if digest!="0d52382ad2d604f75ca28195ad0e0f581dde3cc0386ee7ef69bb9045a18bf2fd":raise RuntimeError("C7 source changed")
    begin=source.index("      SUBROUTINE HOWFAR( IRETE )")
    end=source.index("\n      END\n",begin)+len("\n      END\n")
    routine=source[begin:end].replace('#include "corsika.h"',"""
      DOUBLE PRECISION C(50),BOUND(6),X(1),Y(1),Z(1),U(1),V(1),
     * W(1),WA(1),DNEAR(1),USTEP,OBSLVL(1),WCUT,PRMPAR(20)
      INTEGER NP,IR(1),IOBS(1),IRNEW,NEWOBS,IDISC,NOBSLV,
     * MDEBUG,MONIOU
      LOGICAL FEGSDB,FFLATOUT
      COMMON /GEOD/ C,BOUND,X,Y,Z,U,V,W,WA,DNEAR,USTEP,
     * OBSLVL,WCUT,PRMPAR
      COMMON /GEOI/ NP,IR,IOBS,IRNEW,NEWOBS,IDISC,NOBSLV,
     * MDEBUG,MONIOU
      COMMON /GEOL/ FEGSDB,FFLATOUT
""")
    bodies=[]
    for curved in [0,1]:
        text=subprocess.run(["cpp","-P","-traditional-cpp","-D__UPWARD__=1",f"-D__CURVED__={curved}"],
                            input=routine,text=True,capture_output=True,check=True).stdout
        bodies.append(text.replace("SUBROUTINE HOWFAR(",f"SUBROUTINE HOWFAR{curved}("))
    program="""      PROGRAM REFERENCE
      IMPLICIT NONE
      DOUBLE PRECISION C(50),BOUND(6),X(1),Y(1),Z(1),U(1),V(1),
     * W(1),WA(1),DNEAR(1),USTEP,OBSLVL(1),WCUT,PRMPAR(20),
     * Q(25),OUT(6),THICKNESS
      INTEGER NP,IR(1),IOBS(1),IRNEW,NEWOBS,IDISC,NOBSLV,
     * MDEBUG,MONIOU,I,N
      LOGICAL FEGSDB,FFLATOUT,IRETE
      COMMON /GEOD/ C,BOUND,X,Y,Z,U,V,W,WA,DNEAR,USTEP,
     * OBSLVL,WCUT,PRMPAR
      COMMON /GEOI/ NP,IR,IOBS,IRNEW,NEWOBS,IDISC,NOBSLV,
     * MDEBUG,MONIOU
      COMMON /GEOL/ FEGSDB,FFLATOUT
      COMMON /DEPTH/ THICKNESS
      CHARACTER*1024 INPUT,OUTPUT
      CALL GET_COMMAND_ARGUMENT(1,INPUT)
      CALL GET_COMMAND_ARGUMENT(2,OUTPUT)
      OPEN(10,FILE=TRIM(INPUT),STATUS='OLD')
      OPEN(11,FILE=TRIM(OUTPUT),STATUS='REPLACE')
      NP=1
      NOBSLV=1
      FEGSDB=.FALSE.
      MDEBUG=6
      MONIOU=6
      WCUT=-1.D0
      PRMPAR=0.D0
      READ(10,*) N
      DO I=1,N
        READ(10,*) Q
        FFLATOUT=Q(2).NE.0.D0
        C(1)=Q(3)
        BOUND=Q(4:9)
        OBSLVL(1)=Q(10)
        C(2)=Q(11)
        C(3)=Q(12)
        C(4)=Q(13)
        X(1)=Q(14)
        Y(1)=Q(15)
        Z(1)=Q(16)
        U(1)=Q(17)
        V(1)=Q(18)
        W(1)=Q(19)
        IR(1)=NINT(Q(20))
        IRNEW=IR(1)
        WA(1)=Q(21)
        THICKNESS=Q(22)
        USTEP=Q(23)
        DNEAR(1)=Q(24)
        IOBS(1)=1
        NEWOBS=1
        IDISC=0
        IRETE=.FALSE.
        IF(Q(1).EQ.0.D0)THEN
          CALL HOWFAR0(IRETE)
        ELSE
          CALL HOWFAR1(IRETE)
        ENDIF
        OUT=(/USTEP,DNEAR(1),DBLE(IRNEW),DBLE(NEWOBS),
     *    DBLE(IDISC),0.D0/)
        IF(IRETE)OUT(6)=1.D0
        WRITE(11,'(6(ES26.17E3,1X))') OUT
      ENDDO
      END
      SUBROUTINE AUSGB2
      END
      DOUBLE PRECISION FUNCTION THICK(H)
      IMPLICIT NONE
      DOUBLE PRECISION H,THICKNESS
      COMMON /DEPTH/ THICKNESS
      THICK=THICKNESS
      END
"""+"\n".join(bodies)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/"geometry_oracle.f").write_text(program)
    (a.output/"geometry_oracle_provenance.json").write_text(json.dumps({
        "source":str(a.source.resolve()),"sha256":digest,"not_linked_to_cpp":True,
        "routines":["complete HOWFAR, flat/UPWARD","complete HOWFAR, CURVED/UPWARD"],
        "scope":"Single downward-shower observation level; supplied THICK at current position; not the atmosphere density model"
    },indent=2)+"\n")


if __name__=="__main__":main()
