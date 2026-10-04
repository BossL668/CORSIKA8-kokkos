"""Extract the normal-air flat-frame ELECTR step as a separate reference.

Original HATCH input formats/REAL conversion, ELECTR arithmetic and MSCAT
are retained. Geometry supplies a prescribed cap; HOWFAR, curved rebasing,
optional electric fields, callbacks and longitudinal histograms are outside
this fixture. The executable is never linked to the C++ backend.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("source",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    source=args.source.read_text()
    digest=hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest!="0d52382ad2d604f75ca28195ad0e0f581dde3cc0386ee7ef69bb9045a18bf2fd":
        raise RuntimeError("C7 source changed; re-audit transport extraction")
    def between(a,b,start=0):
        i=source.index(a,start)
        return source[i:source.index(b,i)]
    common="""      INTEGER NP,LELEC,LELKE,NOSCAT,NPTH,NBGB
      REAL RHO,RLC,AE,AP,UE,UP,XR0,TEFF0,BLCC,XCC,EKE0,EKE1,
     * ESIG0(500),ESIG1(500),PSIG0(500),PSIG1(500),
     * EDEDX0(500),EDEDX1(500),PDEDX0(500),PDEDX1(500),
     * EBR10(500),EBR11(500),EBR20(500),EBR21(500),
     * PBR10(500),PBR11(500),PBR20(500),PBR21(500),
     * PBR30(500),PBR31(500),TMXS0(500),TMXS1(500)
      DOUBLE PRECISION E(1),X(1),Y(1),Z(1),U(1),V(1),W(1),TIM(1),
     * PEIE,PRM,RMSQ,RLDU,RLDUI,RHOR(1),HBAROI(1),ECUT(1),
     * STEPFC,STERNCOR,DEMFP,SIG0,BNORM,SINB,COSB,BLIMIT,VACDST,
     * RHOFAC,BETA2,EOLD,TVSTEP,SINTHE,COSTHE,THETA,RD(4),PI,
     * B0G21,B1G21,B0G22,B1G22,B0G31,B1G31,B0G32,B1G32,
     * G210(7),G211(7),G212(7),G220(8),G221(8),G222(8),
     * G310(11),G311(11),G312(11),G320(25),G321(25),G322(25),
     * B0BGB,B1BGB,BGB0(8),BGB1(8),BGB2(8),
     * B0PTH,B1PTH,PTH0(6),PTH1(6),PTH2(6),
     * TWOPI,VCI,CAPF,OUT(18)
      INTEGER*8 RANDOM_STATE
      INTEGER NUSED
"""
    data=between("      DATA B0G21/", "\n      END",source.index("      DATA MED/6*1/"))
    module="      MODULE C7STATE\n      IMPLICIT NONE\n"+common+data+"\n      END MODULE\n"
    local=between("        RHOFAC = RHOR(IRL)/RHO","      ENDIF\n      IRNEW  = IR(NP)",source.index(" 451  CONTINUE"))
    correction=between("      VSTEP = USTEP\n      IF ( USTEP .EQ. USTEP0 ) THEN","C  NOW TAKE IONIZATION LOSSES",source.index("C  WE ARE IN NORMAL MEDIUM WITH NORMAL STEP"))
    a=correction.index("C  KILL UPWARD GOING PARTICLES")
    b=correction.index("C  PATH LENGTH CORRECTION FOR BAROMETRIC AIR",a)
    correction=correction[:a]+correction[b:]
    correction=re.sub(r"          IF \( DEBUG \) THEN\n.*?          ENDIF\n","",correction,count=1,flags=re.S)
    magnetic=between("C  NOW TAKE IONIZATION LOSSES","#if __EFIELD__",source.index("C  WE ARE IN NORMAL MEDIUM WITH NORMAL STEP"))
    normalize=between("C  MAGNETIC DEFLECTION IS APPROXIMATION","#if __EFIELD__",source.index("C  WE ARE IN NORMAL MEDIUM WITH NORMAL STEP"))
    move=between("      X(NP)  = X(NP) + VSTEP*UMEAN","#if __CURVED__",source.index("C  MAGNETIC DEFLECTION IS APPROXIMATION"))
    rotation=between("C  NOW ADD ANGLE OF MULTIPLE SCATTERING","C  UPDATE ENERGY",source.index("C  WE ARE IN NORMAL MEDIUM WITH NORMAL STEP"))
    body="""      SUBROUTINE STEP_REFERENCE
      USE C7STATE
      IMPLICIT NONE
      INTEGER IRL,IPTH
      DOUBLE PRECISION EKE,ELKE,RHOFI,SIG,TSTEP,TMXS,TUSTEP,
     * DEDX0,DEDX,RANGEL,BETA3,TSCAT,RATIO,USTEP,USTEPU,ALTEXP,
     * DISC,TUSTPC,USTEP0,VSTEP,TVSTPC,VSTEPU,VSTP,PTH,ALPHA,
     * DE,EDEP,EKEF,ENEW,U0,V0,W0,FNORM,F1SIN,F1COS,V1,USW,
     * RADINV,UMEAN,VMEAN,WMEAN,PHI,SINPHI,COSPHI,A,B,CC,
     * SINPS2,SINPSI,US,VS,SINDEL,COSDEL,PROPOSED
      IRL=1
      PEIE=E(1)
      EKE=PEIE-PRM
      ELKE=LOG(EKE)
      LELKE=EKE1*ELKE+EKE0
"""+local+"""      USTEP0=USTEP
      USTEP=MIN(USTEP,BLIMIT*PEIE)
      PROPOSED=USTEP
C  Prescribed geometry cap; this is not a replacement HOWFAR.
      USTEP=USTEP*CAPF
"""+correction+magnetic+normalize+move+"""      TIM(NP) = TIM(NP) + TVSTPC*VCI/SQRT(
     *                    (1.D0-(PRM/E(NP)))*(1.D0+(PRM/E(NP))) )
"""+rotation+"""      PEIE=PEIE-EDEP
      E(NP)=PEIE
      IF(PEIE.GT.ECUT(1))
     *  DEMFP=MAX(0.D0,DEMFP-TVSTEP*SIG)
      OUT=(/USTEP0,PROPOSED,VSTEP,TVSTEP,TVSTPC,EDEP,E(1),
     * X(1),Y(1),Z(1),U(1),V(1),W(1),TIM(1),THETA,ALPHA,
     * DEMFP,DEDX/)
      END
"""
    mscat=between("      SUBROUTINE MSCAT\n","\n      END\n")+"\n      END\n"
    mscat=mscat.replace("      IMPLICIT NONE","      USE C7STATE\n      IMPLICIT NONE",1)
    mscat=re.sub(r'#define .*?\n#include "corsika.h"\n',"",mscat,count=1,flags=re.S)
    mscat=re.sub(r"      IF \( FEGSDB \) THEN\n.*?      ENDIF\n","",mscat,count=1,flags=re.S)
    # Keep original clamp; omit a diagnostic WRITE whose unit is not in fixture.
    mscat=re.sub(r"          WRITE\(KMPO,940\) IB\n 940    .*?\n","",mscat,count=1)
    if "KMPO" in mscat:raise RuntimeError("Unexpected MSCAT output dependency")
    reads=between("      READ(KMPI,520) XR0,TEFF0,BLCC,XCC","C  PHOTIN",source.index("      SUBROUTINE HATCH"))
    driver="""      PROGRAM REFERENCE
      USE C7STATE
      IMPLICIT NONE
      INTEGER N,I,J,K,NEKE,KMPI,POS
      REAL SKIP36(36),SKIP7(7)
      DOUBLE PRECISION EI,LI,DFACT,DFACTI
      CHARACTER*1024 INPUT,OUTPUT,EGSDAT,LINE
      CALL GET_COMMAND_ARGUMENT(1,INPUT)
      CALL GET_COMMAND_ARGUMENT(2,OUTPUT)
      CALL GET_COMMAND_ARGUMENT(3,EGSDAT)
      OPEN(10,FILE=TRIM(INPUT),STATUS='OLD')
      OPEN(11,FILE=TRIM(OUTPUT),STATUS='REPLACE')
      KMPI=12
      OPEN(KMPI,FILE=TRIM(EGSDAT),STATUS='OLD')
      READ(KMPI,'(A)') LINE
      READ(KMPI,'(A)') LINE
      POS=INDEX(LINE,'RHO=')
      READ(LINE(POS+4:),*) RHO
      DO I=1,4
        READ(KMPI,'(A)') LINE
      ENDDO
      READ(KMPI,520) RLC,AE,AP,UE,UP
      READ(KMPI,'(A)') LINE
      READ(KMPI,520) SKIP36
      READ(KMPI,520) SKIP7
      NEKE=500
"""+reads+""" 520  FORMAT(1X,1P,5E14.5)
      CLOSE(KMPI)
      DFACT=DBLE(RLC)
      DFACTI=1.D0/DFACT
      DO I=1,NEKE
        ESIG0(I)=ESIG0(I)*DFACTI
        ESIG1(I)=ESIG1(I)*DFACTI
        PSIG0(I)=PSIG0(I)*DFACTI
        PSIG1(I)=PSIG1(I)*DFACTI
        EDEDX0(I)=EDEDX0(I)*DFACTI
        EDEDX1(I)=EDEDX1(I)*DFACTI
        PDEDX0(I)=PDEDX0(I)*DFACTI
        PDEDX1(I)=PDEDX1(I)*DFACTI
        TMXS0(I)=TMXS0(I)*DFACT
        TMXS1(I)=TMXS1(I)*DFACT
      ENDDO
      TEFF0=TEFF0*DFACT
      BLCC=BLCC*DFACTI
      XCC=XCC*SQRT(DFACTI)
      RLDU=RLC
      RLDUI=1.D0/RLDU
      PI=ACOS(-1.D0)
      TWOPI=2.D0*PI
      VCI=1.D0/2.99792458D10
      VACDST=1.D9
      READ(10,*) N
      DO I=1,N
        READ(10,*) LELEC,E(1),PRM,EI,RHOR(1),HBAROI(1),
     *    STERNCOR,ECUT(1),STEPFC,DEMFP,X(1),Y(1),Z(1),
     *    U(1),V(1),W(1),BNORM,SINB,COSB,BLIMIT,CAPF,RANDOM_STATE
        NP=1
        NUSED=0
        NOSCAT=0
        RMSQ=PRM*PRM
        TIM=0.D0
        LI=LOG(EI-PRM)
        J=EKE1*LI+EKE0
        IF(LELEC.LT.0)THEN
          SIG0=ESIG1(J)*LI+ESIG0(J)
        ELSE
          SIG0=PSIG1(J)*LI+PSIG0(J)
        ENDIF
        CALL STEP_REFERENCE
        WRITE(11,'(I8,1X,18(ES26.17E3,1X))') NUSED,OUT
      ENDDO
      END

      SUBROUTINE RMMARD(RANDOMS,N,ISEQ)
      USE C7STATE, ONLY: RANDOM_STATE,NUSED
      IMPLICIT NONE
      INTEGER N,ISEQ,I
      DOUBLE PRECISION RANDOMS(N)
      DO I=1,N
        IF(NUSED.GE.100000) STOP 3
        RANDOM_STATE=MOD(RANDOM_STATE*48271_8,2147483647_8)
        RANDOMS(I)=DBLE(RANDOM_STATE)/2147483647.D0
        NUSED=NUSED+1
      ENDDO
      END
"""
    full=module+driver+body+mscat
    if "#" in full:raise RuntimeError("Unexpected preprocessor branch in extracted local step")
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"transport_oracle.f").write_text(full)
    (args.output/"transport_oracle_provenance.json").write_text(json.dumps({
      "sha256":digest,"source":str(args.source.resolve()),
      "routines":["HATCH electron READ and REAL conversion","ELECTR normal-air local step arithmetic","MSCAT","EGSIN path/scattering DATA"],
      "not_linked_to_cpp":True,
      "scope":"local flat EGS frame; prescribed geometry cap, not HOWFAR/curved full transport",
      "changes":["external deterministic RNG","callbacks, histograms, metadata and cut actions omitted",
                 "explicit per-input initial sigma energy and surviving DEMFP",
                 "local units cm/MeV/s; output energy updated before returning"]
    },indent=2)+"\n")


if __name__=="__main__":main()
