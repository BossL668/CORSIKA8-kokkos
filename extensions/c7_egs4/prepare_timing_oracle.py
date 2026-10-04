"""Make standalone timing drivers around the audited, original C7 kernels.

No Fortran object is linked into the C++ backend. Only the existing oracle's
driver is replaced; kernel arithmetic and REAL table conversion are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path


def once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError(f"Unexpected fixture structure: {before!r}")
    return text.replace(before, after)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("oracles", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for kind in ("transport", "radiative"):
        path = a.oracles / f"{kind}_oracle.f"
        original = path.read_text()
        start = original.index("      PROGRAM REFERENCE")
        end = original.index("\n      END\n", start) + len("\n      END\n")
        driver = original[start:end]
        preamble = original[:start]
        kernels = original[end:]
        declaration = "      CHARACTER*1024 INPUT,OUTPUT,EGSDAT,LINE\n"
        driver = once(driver, declaration, declaration + """      INTEGER MODE,ROUND,REP,REPEATS,ROUNDS,Q
      INTEGER*8 C0,C1,CRATE
      DOUBLE PRECISION,ALLOCATABLE::ENERGIES(:),SEEDS(:),ANS(:,:)
      DOUBLE PRECISION SECONDS,CHECKSUM
      CHARACTER*1024 MODEARG,RAW
""")
        driver = once(driver, "      OPEN(11,FILE=TRIM(OUTPUT),STATUS='REPLACE')", """      OPEN(11,FILE=TRIM(OUTPUT),STATUS='NEW')
      CALL GET_COMMAND_ARGUMENT(4,MODEARG)
      READ(MODEARG,*) MODE
      CALL GET_COMMAND_ARGUMENT(5,RAW)
      OPEN(13,FILE=TRIM(RAW),STATUS='NEW',ACCESS='STREAM',
     * FORM='UNFORMATTED')
      ROUNDS=7
      REPEATS=5
      WRITE(11,'(A)') 'round,seconds_per_batch,checksum'
""")
        driver = driver[:driver.index("      READ(10,*) N")]
        driver += """      READ(10,*) N
      ALLOCATE(ENERGIES(N),SEEDS(N),ANS(22,N))
      DO I=1,N
        READ(10,*) ENERGIES(I),SEEDS(I)
      ENDDO
      CLOSE(10)
      CALL SYSTEM_CLOCK(COUNT_RATE=CRATE)
      DO ROUND=-2,ROUNDS-1
        CALL SYSTEM_CLOCK(C0)
        DO REP=1,REPEATS
          DO I=1,N
            ANS(:,I)=0.D0
            RANDOM_STATE=INT(SEEDS(I),8)
            NUSED=0
            NP=1
            PRM=.51099895D0
"""
        if kind == "transport":
            driver += """            E(1)=ENERGIES(I)
            LELEC=MERGE(-1,1,MOD(I,2).EQ.0)
            RMSQ=PRM*PRM
            NOSCAT=0
            TIM=0.D0
            IF(MODE.EQ.2)THEN
              EOLD=E(1)
              BETA2=MAX(1.D-8,1.D0-RMSQ/EOLD**2)
              RHOFAC=1.D0
              TVSTEP=.01D0/DBLE(RHO)
              CALL MSCAT
              ANS(5,I)=THETA
              ANS(2,I)=NOSCAT
            ELSE
              RHOR(1)=.001225D0
              HBAROI(1)=1.25D-6
              STERNCOR=10.D0
              ECUT(1)=PRM+.5D0
              STEPFC=1.D0
              IF(MODE.EQ.1) STEPFC=.0625D0
              DEMFP=.1D0
              X(1)=10.D0
              Y(1)=-20.D0
              Z(1)=-1.D6
              U(1)=.3D0
              V(1)=.4D0
              W(1)=SQRT(.75D0)
              BNORM=5.D-5*2.99792458D0
              SINB=.6D0
              COSB=.8D0
              BLIMIT=.2D0/BNORM
              CAPF=.8D0
              LI=LOG(E(1)-PRM)
              J=EKE1*LI+EKE0
              IF(LELEC.LT.0)THEN
                SIG0=ESIG1(J)*LI+ESIG0(J)
              ELSE
                SIG0=PSIG1(J)*LI+PSIG0(J)
              ENDIF
              CALL STEP_REFERENCE
              ANS(5:22,I)=OUT
            ENDIF
"""
        else:
            driver += """            E=0.D0
            U=0.D0
            V=0.D0
            W=0.D0
            IQ=0
            Z=-110000.D0
            E(1)=ENERGIES(I)
            U(1)=.3D0
            V(1)=.4D0
            W(1)=SQRT(.75D0)
            FROZEN_DENSITY=.001225D0
            RMI=1.D0/PRM
            FPASS=.FALSE.
            IF(MODE.EQ.3)THEN
              IQ(1)=MERGE(3,2,MOD(I,2).EQ.0)
              CALL BREMSLPM(FPASS)
            ELSE
              IQ(1)=1
              CALL PAIRLPM(FPASS)
            ENDIF
            IF(FPASS.OR.NP.NE.2) STOP 4
            ANS(2,I)=NP
            ANS(3,I)=IQ(1)
            ANS(4,I)=IQ(2)
            ANS(5:12,I)=(/E(1),U(1),V(1),W(1),
     *        E(2),U(2),V(2),W(2)/)
"""
        driver += """            ANS(1,I)=NUSED
          ENDDO
        ENDDO
        CALL SYSTEM_CLOCK(C1)
        SECONDS=DBLE(C1-C0)/DBLE(CRATE)/DBLE(REPEATS)
        CHECKSUM=SUM(ANS)/DBLE(N)
        IF(ROUND.GE.0) WRITE(11,'(I3,A,ES26.17,A,ES26.17)')
     *    ROUND,',',SECONDS,',',CHECKSUM
      ENDDO
      WRITE(13) ANS
      CLOSE(13)
      CLOSE(11)
      END
"""
        # Compile driver/kernels separately, without LTO, so timed calls cannot
        # be removed or hoisted across repeated identical batches.
        (a.output / f"{kind}_state.f").write_text(preamble or "C no module\n")
        (a.output / f"{kind}_timing.f").write_text(driver)
        (a.output / f"{kind}_kernels.f").write_text(kernels)
        provenance[kind] = {
            "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "kernels_sha256": hashlib.sha256(kernels.encode()).hexdigest(),
            "kernel_change": "none relative to audited original-C7 oracle",
        }
    (a.output / "timing_oracle_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
