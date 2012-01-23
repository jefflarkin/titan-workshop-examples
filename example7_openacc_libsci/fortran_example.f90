      program example_OpenACC
      integer  m,n,k,lda,ldb,ldc,iseed(4)
      integer  i,j
      double precision, allocatable::  a(:,:), b(:,:), c(:,:), c2(:,:)
      double precision  alpha, beta, work,error, maximu
      double precision  dlange, Wall_Time,flops
      double precision  t1, t2, t_acc, t_cpu
      external dlange,Wall_Time

      allocate(a(4096,4096),b(4096,4096),c(4096,4096),c2(4096,4096))
      alpha = 1.0D+0
      beta =  0.0D+0
      iseed(1)=0
      iseed(2)=0
      iseed(3)=0
      iseed(4)=1
      m =4096
      n =4096
      k =4096
      lda =4096
      ldb =4096
      ldc =4096
!
!     Create random matrices
!
      call dlarnv( 1, iseed, 4096*4096, a )
      call dlarnv( 1, iseed, 4096*4096, b )
      call dlarnv( 1, iseed, 4096*4096, c )
      c2(:,:) = c(:,:)

!
!     Call GPU routine
!

      t1 = Wall_Time()
!$acc data copy(a,b,c)
!$acc host_data use_device(a,b,c)
      Call dgemm_acc('n','n',m,n,k,alpha,a,lda,b,ldb,beta,c,ldc)
!$acc end host_data
!$acc end data
      t2 = Wall_Time()
      t_acc = t2-t1

!
!     Call CPU routine
!
      t1 = Wall_Time()
      call dgemm_cpu('n','n',m,n,k,alpha,a,lda,b,ldb,beta,c2,ldc)
      t2 = Wall_Time()
      t_cpu = t2-t1
!
!     Get the maximum error
!
      c(:,:) = c(:,:)-c2(:,:)
      error = dlange('M',m,n,c,ldc,work)
      flops = dble(m)*dble(n)*dble(k)
      print *,'Error', error
      print *, 'DGEMM_ACC: ', (2.0*flops)*1.0d-9/(t_acc),'GFlops'
      print *, 'DGEMM_CPU: ', (2.0*flops)*1.0d-9/(t_cpu),'GFlops'
      deallocate(a)
      deallocate(b)
      deallocate(c)
      deallocate(c2)
      end program

