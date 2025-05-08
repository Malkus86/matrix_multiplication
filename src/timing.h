/**
 * High-precision timing utility for performance measurement
 */

 #ifndef TIMING_H
 #define TIMING_H
 
 #include <windows.h>
 
 /**
  * Get current time with high precision
  * @return Time in seconds as a double
  */
 double get_time() {
     LARGE_INTEGER frequency;
     LARGE_INTEGER start;
     QueryPerformanceFrequency(&frequency);
     QueryPerformanceCounter(&start);
     return (double)start.QuadPart / (double)frequency.QuadPart;
 }
 
 #endif // TIMING_H