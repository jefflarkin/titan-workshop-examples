program scaleit
  use mpi
  external scaleit_launcher
  integer scaleit_launcher

  integer :: i,rank,ierr
  integer,parameter :: n=16384
  real(8) :: a(n)

  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)

  !$acc data copyout(a)
  !$acc parallel loop
  do i=1,n
    a(i) = i
  enddo
  !$acc end parallel loop

  !$acc host_data use_device(a)
  ierr = scaleit_launcher(a, n, rank)
  !$acc end host_data
  !$acc end data

  do i=1,n
    if(a(i).ne.(real(rank)*i)) then
      write(*,*)"Error",i,a(i)
      call MPI_Abort(MPI_COMM_WORLD,-1, ierr)
    endif
  enddo
  write(*,*)"Correct"

  call mpi_finalize()
end program
