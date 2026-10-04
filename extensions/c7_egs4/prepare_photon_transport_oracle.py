"""Separate original PHOTON normal-air arithmetic fixture, not a linked backend.

Preserves HATCH REAL photon tables and conversion, the min-distance rules,
barometric map, prescribed geometric cap and flat-frame motion. No HOWFAR,
Rayleigh scattering, curved rebasing, observation callbacks or shower loop.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    source=args.source.read_text()
    digest=hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest!="0d52382ad2d604f75ca28195ad0e0f581dde3cc0386ee7ef69bb9045a18bf2fd":
        raise RuntimeError("C7 source changed; re-audit photon transport")
    begin=source.index("      SUBROUTINE PHOTON( IRCODE )")
    def block(a,b,start=begin):
        pos=source.index(a,start)
        return source[pos:source.index(b,pos)]
    proposal=block("        RHOFAC = RHOR(IRL)/RHO", "      ENDIF\n      IRNEW  = IR(NP)")
    # The reader/model explicitly reject Rayleigh-enabled tables.
    ray_start=proposal.index("C  DENSITY CORRECTION OF MEAN FREE PATH")
    ray_end=proposal.index("        TSTEP  = MAX",ray_start)
    proposal=proposal[:ray_start]+proposal[ray_end:]
    optical=block("      VSTEP  = USTEP\n      TVSTEP = VSTEP", "#if __UPWARD__",begin)
    move=block("      X(NP)   = X(NP)+U(NP)*USTEP", "#if __CURVED__",begin)
    photon_reads=block("      READ(KMPI,520) EBINDA,GE0,GE1", "      IF ( IRAYLM",source.index("      SUBROUTINE HATCH"))
    program="""      PROGRAM REFERENCE
      IMPLICIT NONE
      INTEGER I,N,NP,IRL,KMPI,NGE,LGLE
      REAL RHO,RLC,AE,AP,UE,UP,SKIP36(36),SKIP7(7),SKIP4(4),
     * SKIP2(2),SKIP10000(10000),EBINDA,GE0,GE1,
     * GMFP0(500),GMFP1(500),GBR10(500),GBR11(500),
     * GBR20(500),GBR21(500),GBR30(500),GBR31(500),
     * GBR40(500),GBR41(500)
      DOUBLE PRECISION E(1),X(1),Y(1),Z(1),U(1),V(1),W(1),
     * TIM(1),RHOR(1),HBAROI(1),DPMFP,GMFPR0,GMFP,RHOFAC,
     * RHOFI,TSTEP,ALTEXP,DISC,VACDST,CAP,USTEP,VSTEP,TVSTEP,
     * EDEP,USTEPU,GLE,VCI,DFACT,OUT(15)
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
      I=INDEX(LINE,'RHO=')
      READ(LINE(I+4:),*) RHO
      DO I=1,4
        READ(KMPI,'(A)') LINE
      ENDDO
      READ(KMPI,520) RLC,AE,AP,UE,UP
      READ(KMPI,'(A)') LINE
      READ(KMPI,520) SKIP36
      READ(KMPI,520) SKIP7
      READ(KMPI,520) SKIP4
      READ(KMPI,520) SKIP2
      READ(KMPI,520) SKIP10000
      NGE=500
"""+photon_reads+""" 520  FORMAT(1X,1P,5E14.5)
      CLOSE(KMPI)
      DFACT=DBLE(RLC)
      DO I=1,NGE
        GMFP0(I)=GMFP0(I)*DFACT
        GMFP1(I)=GMFP1(I)*DFACT
      ENDDO
      VCI=1.D0/2.99792458D10
      VACDST=1.D9
      NP=1
      IRL=1
      READ(10,*) N
      DO I=1,N
        READ(10,*) E(1),X(1),Y(1),Z(1),U(1),V(1),W(1),
     *   RHOR(1),HBAROI(1),DPMFP,CAP
        TIM=0.D0
        GLE=LOG(E(1))
        LGLE=GE1*GLE+GE0
        GMFPR0=GMFP1(LGLE)*GLE+GMFP0(LGLE)
"""+proposal+"""        USTEP=TSTEP*CAP
"""+optical+move+"""        TIM(NP)=TIM(NP)+TVSTEP*VCI
        DPMFP=MAX(0.D0,DPMFP-USTEPU/GMFP)
        OUT=(/GMFPR0,GMFP,TSTEP,USTEP,USTEPU,X(1),Y(1),Z(1),
     *    TIM(1),DPMFP,E(1),GBR11(LGLE)*GLE+GBR10(LGLE),
     *    GBR21(LGLE)*GLE+GBR20(LGLE),GBR31(LGLE)*GLE+GBR30(LGLE),
     *    GBR41(LGLE)*GLE+GBR40(LGLE)/)
        WRITE(11,'(15(ES26.17E3,1X))') OUT
      ENDDO
      END
"""
    if "#" in program:raise RuntimeError("Unexpected preprocessor branch")
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"photon_transport_oracle.f").write_text(program)
    (args.output/"photon_transport_oracle_provenance.json").write_text(json.dumps({
        "source":str(args.source.resolve()),"sha256":digest,"not_linked_to_cpp":True,
        "routines":["HATCH photon READ/REAL conversion","PHOTON normal-air proposal and motion"],
        "scope":"flat local frame, supplied geometry cap; no Rayleigh, curved geometry or full shower"
    },indent=2)+"\n")


if __name__=="__main__":main()
