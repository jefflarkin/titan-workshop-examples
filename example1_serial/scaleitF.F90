! CUDA Fortran requires CUDA code to be in a Fortran module
module scaleit_mod
  use cudafor
  implicit none
  private

  public scaleit_simple, scaleit_complex

  contains
  
  ! Device Kernel
  attributes(global)&
  subroutine scaleit_kernel(a,n)
    real(8),intent(inout)    :: a(n)
    integer,intent(in),value :: n
    integer i

    i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (i.le.n) then
      a(i) = 2.0 * a(i)
    endif
  end subroutine scaleit_kernel

  ! Kernel Launcher, Simple
  subroutine scaleit_simple(h_a,n)
    real(8),intent(inout) :: h_a(n)
    ! Declare d_a as a device array
    real(8),device        :: d_a(n)
    integer,intent(in)    :: n
    type(dim3)            :: blk, grd

    ! Decompose the problem
    blk = dim3(1024,1,1)
    grd = dim3(n/blk%x,1,1)

    ! Copy from Host to Device
    d_a = h_a

    ! Launch the compute kernel
    call scaleit_kernel<<<grd,blk>>>(d_a,n)

    ! Copy from Device to Host
    h_a = d_a
  end subroutine scaleit_simple

  ! Kernel Launcher, Comples
  subroutine scaleit_complex(h_a,n)
    real(8),intent(inout)      :: h_a(n)
    ! Declare d_a as a device array
    real(8),device,allocatable :: d_a(:)
    integer,intent(in)         :: n
    type(dim3)                 :: blk, grd
    integer                    :: ierr

    ! Allocate device array
    allocate(d_a(size(h_a)))

    ! Decompose the problem
    blk = dim3(1024,1,1)
    grd = dim3(n/blk%x,1,1)

    ! Copy from Host to Device
    ! d_a = a
    ierr = cudaMemcpy(d_a,h_a,size(h_a),cudaMemcpyHostToDevice)

    ! Launch the compute kernel
    call scaleit_kernel<<<grd,blk>>>(d_a,n)

    ! Copy from Device to Host
    ! a = d_a
    ierr = cudaMemcpy(h_a,d_a,size(h_a),cudaMemcpyDeviceToHost)

    ! Deallocate device array
    deallocate(d_a)
  end subroutine scaleit_complex
end module scaleit_mod

program scaleit
  use scaleit_mod
  integer, parameter :: n = 16384
  real(8) :: a(n)
  integer :: i

  do i=1,n
    a(i) = i
  enddo

  call scaleit_simple(a,n)

  do i=1,n
    if(a(i).ne.(2.0*i)) then
      write(*,*)"Error",i,a(i)
      STOP
    endif
  enddo

  call scaleit_complex(a,n)

  do i=1,n
    if(a(i).ne.(4.0*i)) then
      write(*,*)"Error",i,a(i)
      STOP
    endif
  enddo

  write(*,*)"Correct"
end program scaleit
