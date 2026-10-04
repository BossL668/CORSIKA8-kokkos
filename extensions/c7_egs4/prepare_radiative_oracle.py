"""Build-only extraction of BREMSLPM/PAIRLPM/LPMEFFECT/UPHI for an oracle.

The native backend never links this executable. HATCH's BREMPR reads are
retained independently from the C++ reader. RHOF returns a supplied density
for these local-kernel checks; atmospheric geometry is not validated here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("source",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    source=args.source.read_text()
    digest=hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest!="0d52382ad2d604f75ca28195ad0e0f581dde3cc0386ee7ef69bb9045a18bf2fd":
        raise RuntimeError("C7 source hash changed")
    common="""      INTEGER NP,IQ(2)
      DOUBLE PRECISION E(2),U(2),V(2),W(2),Z(2),PRM,RMI,
     * RD(4),TWOPI,PI,SINTHE,COSTHE,SINPHI,COSPHI,THETA,API
      COMMON /C7STATE/ E,U,V,W,Z,PRM,RMI,RD,TWOPI,PI,
     * SINTHE,COSTHE,SINPHI,COSPHI,THETA,API,NP,IQ
      DOUBLE PRECISION PWR2I(60)
      REAL DL1(6),DL2(6),DL3(6),DL4(6),DL5(6),DL6(6),
     * DELCM,ALPHI(2),BPAR(2),DELPOS(2),AP
      COMMON /COEFFICIENTS/ PWR2I,DL1,DL2,DL3,DL4,DL5,DL6,
     * DELCM,ALPHI,BPAR,DELPOS,AP
      DOUBLE PRECISION FROZEN_DENSITY
      COMMON /ATMOSPHERE/ FROZEN_DENSITY
      INTEGER*8 RANDOM_STATE
      INTEGER NUSED
      COMMON /RANDOMSTATE/ RANDOM_STATE,NUSED
"""
    routines=[]
    for name in ("BREMSLPM", "PAIRLPM", "LPMEFFECT", "UPHI"):
        begin=source.index("      SUBROUTINE "+name+"(")
        end=source.index("\n      END\n",begin)+len("\n      END\n")
        part=source[begin:end]
        if name in ("BREMSLPM","PAIRLPM"):
            part=re.sub(r"\n#else\n      SUBROUTINE \w+\n#endif", "", part,count=1)
        part=re.sub(r'#define .*?\n#include "corsika.h"\n',common,part,count=1,flags=re.S)
        part=re.sub(r"      IF \( FEGSDB \) THEN\n.*?      ENDIF\n","",part,count=1,flags=re.S)
        part=re.sub(r"      IGEN\(NP(?:-1)?\) = IGEN\(NP(?:-1)?\) \+ (?:1000|1000000)\n","",part)
        if name=="LPMEFFECT":
            part=re.sub(r"        IF \( FEGSDB \)\n.*?^ 1      FORMAT\(.*?\n","",part,count=1,flags=re.S|re.M)
            part=part.replace("        IF ( FEGSDB ) WRITE(MDEBUG,*) 'LPMEFF: FPASS= ',FPASS\n","")
        if name=="UPHI":
            a=part.index("      X(NP) = X(NP-1)")
            b=part.index(" 1130 CONTINUE",a)
            part=part[:a]+part[b:]
        part=subprocess.run(["cpp","-P","-traditional-cpp","-D__LPM__=1"],
                            input=part,text=True,capture_output=True,check=True).stdout
        if "#" in part or re.search(r"\b(FEGSDB|IGEN)\b",part):
            raise RuntimeError("Unremoved diagnostic code: "+name)
        routines.append(part)
    driver="""      PROGRAM REFERENCE
      IMPLICIT NONE
"""+common+"""      INTEGER N,I,J,PID
      REAL RLC,AE,UE,UP
      DOUBLE PRECISION P
      LOGICAL FPASS
      CHARACTER*1024 INPUT,OUTPUT,EGSDAT,LINE
      CALL GET_COMMAND_ARGUMENT(1,INPUT)
      CALL GET_COMMAND_ARGUMENT(2,OUTPUT)
      CALL GET_COMMAND_ARGUMENT(3,EGSDAT)
      OPEN(10,FILE=TRIM(INPUT),STATUS='OLD')
      OPEN(11,FILE=TRIM(OUTPUT),STATUS='REPLACE')
      OPEN(12,FILE=TRIM(EGSDAT),STATUS='OLD')
C  Read the current 4-element AIR-NTP medium. Preserve HATCH READ formats.
      DO I=1,6
        READ(12,'(A)') LINE
      ENDDO
      READ(12,520) RLC,AE,AP,UE,UP
      READ(12,'(A)') LINE
      READ(12,520) (DL1(I),DL2(I),DL3(I),DL4(I),DL5(I),
     * DL6(I),I=1,6)
      READ(12,520) DELCM,(ALPHI(I),BPAR(I),DELPOS(I),I=1,2)
 520  FORMAT(1X,1P,5E14.5)
      CLOSE(12)
      API=1.D0/AP
      P=1.D0
      DO I=1,60
        PWR2I(I)=P
        P=P*.5D0
      ENDDO
      PI=ACOS(-1.D0)
      TWOPI=2.D0*PI
      READ(10,*) N
      DO I=1,N
        E=0.D0
        U=0.D0
        V=0.D0
        W=0.D0
        Z=-110000.D0
        IQ=0
        READ(10,*) PID,E(1),PRM,FROZEN_DENSITY,
     *    U(1),V(1),W(1),RANDOM_STATE
        NP=1
        NUSED=0
        RMI=1.D0/PRM
        FPASS=.FALSE.
        IF(PID.EQ.22) THEN
          IQ(1)=1
          CALL PAIRLPM(FPASS)
        ELSEIF(PID.EQ.11.OR.PID.EQ.-11) THEN
          IQ(1)=3
          IF(PID.EQ.-11) IQ(1)=2
          CALL BREMSLPM(FPASS)
        ELSE
          STOP 2
        ENDIF
        WRITE(11,'(5(I8,1X),8(ES26.17E3,1X))')
     *  NP,NUSED,IQ(1),IQ(2),MERGE(1,0,FPASS),
     *  E(1),U(1),V(1),W(1),E(2),U(2),V(2),W(2)
      ENDDO
      CLOSE(10)
      CLOSE(11)
      END

      SUBROUTINE RMMARD(RANDOMS,N,ISEQ)
      IMPLICIT NONE
      INTEGER N,ISEQ,NUSED,J
      INTEGER*8 RANDOM_STATE
      DOUBLE PRECISION RANDOMS(N)
      COMMON /RANDOMSTATE/ RANDOM_STATE,NUSED
C  Test-only deterministic random provider, shared input seed, no overflow.
      DO J=1,N
        RANDOM_STATE=MOD(RANDOM_STATE*48271_8,2147483647_8)
        RANDOMS(J)=DBLE(RANDOM_STATE)/2147483647.D0
      ENDDO
      NUSED=NUSED+N
      IF(NUSED.GT.100000) STOP 3
      END

      DOUBLE PRECISION FUNCTION RHOF(ALT)
      IMPLICIT NONE
      DOUBLE PRECISION ALT,FROZEN_DENSITY
      COMMON /ATMOSPHERE/ FROZEN_DENSITY
      RHOF=FROZEN_DENSITY
      END
"""
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"radiative_oracle.f").write_text(driver+"\n".join(routines))
    (args.output/"radiative_oracle_provenance.json").write_text(json.dumps({
        "source":str(args.source.resolve()),"source_sha256":digest,
        "routines":["BREMSLPM","PAIRLPM","LPMEFFECT","UPHI"],
        "runtime_dependency":False,
        "changes":["Debug and metadata copies removed", "Original energy/angle/rejection arithmetic retained",
                   "RHOF stub supplies the test density, not an atmospheric geometry test",
                   "Test-only deterministic random provider", "BREMPR read independently using HATCH format"],
    },indent=2)+"\n")


if __name__=="__main__":main()
