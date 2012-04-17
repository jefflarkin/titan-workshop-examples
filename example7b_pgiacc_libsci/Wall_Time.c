/*--------------------------------------------------------*\
  Return the current Wall Time Stamp in seconds
  uses standard gettimeofday function
 ----------------------------------------------------------
 $Id: Wall_Time.c,v 1.1 2005/01/09 23:59:35 rickyk Exp $
\*--------------------------------------------------------*/
#include <sys/time.h>
#include <unistd.h>
double Wall_Time(void)
{
  struct timeval mytp;
  double seconds;
  if (!(gettimeofday(&mytp, (struct timezone *)NULL))) {
    seconds  = (double) mytp.tv_sec;
    seconds += (double) mytp.tv_usec*(double)1.0e-06;
    return seconds;
  }
  else {
    return (double) 911;
  }
}
double wall_time_(void)
{
  return Wall_Time();
}
double wall_time__(void)
{
  return Wall_Time();
}
double wall_time(void)
{
  return Wall_Time();
}
double WALL_TIME(void)
{
  return Wall_Time();
}
