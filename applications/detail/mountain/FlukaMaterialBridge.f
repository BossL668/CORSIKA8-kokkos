C     Mountain-only setup; uses the user's installed FLUKA include files.
C     The auxiliary compound initializes all elemental targets with one
C     region. Its arbitrary weights are NEVER used for shower sampling:
C     CORSIKA selects the returned pure elements using the material card.
      SUBROUTINE C8_MATERIAL_FLUKA_SETUP(N,IZ,MT,STATUS)
     & BIND(C,NAME="c8_material_fluka_setup")
      IMPLICIT NONE
      INCLUDE 'fluka_DIMPAR.inc'
      INCLUDE 'fluka_FLKMAT.inc'
      INCLUDE 'fluka_FLKCMP.inc'
      INTEGER N, IZ(N), MT(N), STATUS, NEL(1), COMPOUND(1), IFL
      INTEGER I,J,K,M
      DOUBLE PRECISION WF(N)
      LOGICAL LPRINT
      CHARACTER*8 CRVRCK
      STATUS=1
      IF(N.LT.1.OR.N.GT.92) RETURN
      IF(ANY(IZ.LT.1).OR.ANY(IZ.GT.92)) RETURN
      NEL(1)=N
      WF=1.D0/N
      IFL=1
      LPRINT=.FALSE.
      CRVRCK='76466879'
      CALL STPXYZ(1,NEL,IZ,WF,N,1.D11,0.D0,-1.D0,IFL,
     &            LPRINT,COMPOUND,CRVRCK)
C     Verify the loaded library/header layout and every target mapping.
      STATUS=2
      J=COMPOUND(1)
      IF(J.LT.1.OR.J.GT.MXXMDF) RETURN
      IF(N.EQ.1) THEN
         IF(ICOMP(J).NE.0) RETURN
         IF(ABS(ZTAR(J)-DBLE(IZ(1))).GT.1.D-12) RETURN
         MT(1)=J
         STATUS=0
         RETURN
      END IF
      IF(ICOMPL(J).NE.N) RETURN
      K=ICOMP(J)
      IF(K.LT.1.OR.K+N-1.GT.ICOMAX) RETURN
      STATUS=3
      DO I=1,N
         M=MATNUM(K+I-1)
         IF(M.LT.1.OR.M.GT.MXXMDF) RETURN
         IF(ICOMP(M).NE.0) RETURN
         IF(ABS(ZTAR(M)-DBLE(IZ(I))).GT.1.D-12) RETURN
         MT(I)=M
      END DO
      STATUS=0
      END
