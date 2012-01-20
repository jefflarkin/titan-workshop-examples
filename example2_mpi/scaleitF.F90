! CUDA Fortran requires CUDA code to be in a Fortran module
module scaleit_mod
  use cudafor
  implicit none
  private

  public scaleit_simple

  contains
  
  ! Device Kernel
  attributes(global)&
  subroutine scaleit_kernel(a,n,by)
    real(8),intent(inout)    :: a(n)
    integer,intent(in),value :: n,by
    integer i

    i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (i.le.n) then
      a(i) = real(by) * a(i)
    endif
  end subroutine scaleit_kernel

  ! Kernel Launcher, Simple
  subroutine scaleit_simple(h_a,n,scaleBy)
    real(8),intent(inout) :: h_a(n)
    ! Declare d_a as a device array
    real(8),device        :: d_a(n)
    integer,intent(in)    :: n,scaleBy
    type(dim3)            :: blk, grd

    ! Decompose the problem
    blk = dim3(1024,1,1)
    grd = dim3(n/blk%x,1,1)

    ! Copy from Host to Device
    d_a = h_a

    ! Launch the compute kernel
    call scaleit_kernel<<<grd,blk>>>(d_a,n,scaleBy)

    ! Copy from Device to Host
    h_a = d_a
  end subroutine scaleit_simple
end module scaleit_mod

program scaleit
  use scaleit_mod
  use mpi
  integer, parameter :: n = 16384
  real(8) :: a(n)
  integer :: i,rank,ierr

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)

  do i=1,n
    a(i) = i
  enddo

  call scaleit_simple(a,n,rank)

  do i=1,n
    if(a(i).ne.(real(rank)*i)) then
      write(*,*)"Error",i,a(i)
      STOP
    endif
  enddo
  write(*,*)"Correct"

  call MPI_Finalize()
end program scaleit
