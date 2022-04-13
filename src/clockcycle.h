#ifndef CLOCKCYCLE_H
#define CLOCKCYCLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus
/*
#ifdef __clang__
  static inline uint64_t clock_now()
  {
    return __builtin_readcyclecounter();
  }
*/
#if defined(__GNUC__)
#if defined(__i386__) || defined(__x86_64__) || defined(__amd64__)
#include <x86intrin.h>
    static inline uint64_t clock_now()
    {
        return __rdtsc();
    }
#else
    uint64_t clock_now(void)
    {
        unsigned int tbl, tbu0, tbu1;

        do
        {
            __asm__ __volatile__("mftbu %0"
                                 : "=r"(tbu0));
            __asm__ __volatile__("mftb %0"
                                 : "=r"(tbl));
            __asm__ __volatile__("mftbu %0"
                                 : "=r"(tbu1));
        } while (tbu0 != tbu1);

        return (((uint64_t)tbu0) << 32) | tbl;
    }
#endif
#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // CLOCKCYCLE_H