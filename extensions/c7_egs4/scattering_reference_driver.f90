! External test executable only. Reuses the earlier B C7 MSCAT reference.
program reference
  use iso_c_binding
  implicit none
  interface
    subroutine oracle(energy,grammage,ns,randoms,theta,nused,istat) bind(C,name="c7_oracle")
      import c_double,c_int
      real(c_double),value :: energy,grammage
      integer(c_int),value :: ns
      real(c_double) :: randoms(ns),theta
      integer(c_int) :: nused,istat
    end subroutine
  end interface
  integer,parameter :: ns=4096
  integer n,i,j,nused,istat
  integer(c_int64_t) state
  real(c_double) energy,grammage,theta,randoms(ns)
  character(1024) input,output
  call get_command_argument(1,input)
  call get_command_argument(2,output)
  open(10,file=trim(input),status='old')
  open(11,file=trim(output),status='replace')
  read(10,*) n
  do i=1,n
    read(10,*) energy,grammage,state
    do j=1,ns
      state=mod(state*48271_c_int64_t,2147483647_c_int64_t)
      randoms(j)=dble(state)/2147483647.d0
    end do
    call oracle(energy,grammage,ns,randoms,theta,nused,istat)
    write(11,'(2(I8,1X),ES26.17E3)') nused,istat,theta
  end do
  close(10)
  close(11)
end program
