#include <cstdio>
#include <cstring>
#include <string>

// Tamariba: no host sockets here. Local wireless goes to the other players of a Tamariba room
// (tamariba_mp.cpp) and the access point's internet traffic leaves through the Tamariba
// Plaza (tamariba_lan.cpp). The old UDP broadcast on port 7064 is gone: in this build its
// socket() was a macro returning 0, so the first game to turn Wi-Fi on closed fd 0.
#include "tamariba_lan.h"
#include "tamariba_mp.h"

#ifdef HAVE_THREADS
#include <stdlib.h>

#include <rthreads/rthreads.h>
#include <rthreads/rsemaphore.h>
#endif

#include <streams/file_stream.h>
#include <streams/file_stream_transforms.h>
#include <retro_timers.h>

#include "types.h"
#include "utils.h"
#include "Platform.h"

extern char retro_base_directory[4096];

namespace Platform
{
    FILE* OpenFile(const char* path, const char* mode, bool mustexist)
    {
    FILE* ret;

    if (mustexist)
    {
        ret = fopen(path, "rb");
        if (ret) fclose(ret);
        ret = fopen(path, mode);
    }
    else
        ret = fopen(path, mode);

    return ret;
}

   FILE* OpenLocalFile(const char* path, const char* mode)
   {
      std::string fullpath = std::string(retro_base_directory) + std::string(1, PLATFORM_DIR_SEPERATOR) + std::string(path);
      FILE* f = OpenFile(fullpath.c_str(), mode, true);
      return f;
   }

   FILE* OpenDataFile(const char* path)
   {
      return OpenLocalFile(path, "rb");
   }

   void StopEmu()
   {
       return;
   }

   void Semaphore_Reset(Semaphore *sema)
   {
   #ifdef HAVE_THREADS
      while (ssem_get((ssem_t*)sema) > 0) {
        ssem_trywait((ssem_t*)sema);
      }
   #endif
   }

   void Semaphore_Post(Semaphore *sema, int count)
   {
   #ifdef HAVE_THREADS
       for (int i = 0; i < count; i++)
      {
         ssem_signal((ssem_t*)sema);
      }
   #endif
   }

   void Semaphore_Wait(Semaphore *sema)
   {
   #ifdef HAVE_THREADS
      ssem_wait((ssem_t*)sema);
   #endif
   }

   void Semaphore_Free(Semaphore *sema)
   {
   #ifdef HAVE_THREADS
      ssem_t *sem = (ssem_t*)sema;
      if (sem)
         ssem_free(sem);
   #endif
   }

   Semaphore *Semaphore_Create()
   {
   #ifdef HAVE_THREADS
      ssem_t *sem = ssem_new(0);
      if (sem)
         return (Semaphore*)sem;
   #endif
      return NULL;
   }

   Mutex* Mutex_Create()
   {
   #ifdef HAVE_THREADS
      return (Mutex*)slock_new();
   #endif
      return NULL;
   }

   void Mutex_Free(Mutex* mutex)
   {
   #ifdef HAVE_THREADS
      slock_free((slock_t*)mutex);
   #endif
   }

   void Mutex_Lock(Mutex* mutex)
   {
   #ifdef HAVE_THREADS
      slock_lock((slock_t*)mutex);
   #endif
   }

   void Mutex_Unlock(Mutex* mutex)
   {
   #ifdef HAVE_THREADS
      slock_unlock((slock_t*)mutex);
   #endif
   }

   bool Mutex_TryLock(Mutex* mutex)
   {
   #ifdef HAVE_THREADS
      slock_try_lock((slock_t*)mutex);
   #endif
      return true;
   }

   void Thread_Free(Thread *thread)
   {
   #if HAVE_THREADS
      sthread_detach((sthread_t*)thread);
   #endif
   }

   struct ThreadData
   {
      std::function<void()> fn;
   };

   void function_trampoline(void* param) {
      ThreadData* data = (ThreadData*)param;
      data->fn();
      delete data;
   }

   Thread *Thread_Create(std::function<void()> func)
   {
   #if HAVE_THREADS
      return (Thread*) sthread_create(function_trampoline, new ThreadData{func});
   #endif
   }

   void Thread_Wait(Thread *thread)
   {
   #if HAVE_THREADS
      sthread_join((sthread_t*)thread);
   #endif
   }


   bool MP_Init()
   {
      return true;
   }

   void MP_DeInit()
   {
   }

   int MP_SendPacket(u8* data, int len)
   {
      return TamaribaMP::Send(data, len, 0);
   }

   int MP_SendPacket(u8* data, int len, u64 timestamp)
   {
      return TamaribaMP::Send(data, len, timestamp);
   }

   int MP_RecvPacket(u8* data, bool block)
   {
      return MP_RecvPacket(data, block, nullptr);
   }

   int MP_RecvPacket(u8* data, bool block, MPFrameInfo* info)
   {
      TamaribaMP::RxInfo rx;
      int len = TamaribaMP::Recv(data, block, &rx);
      if (info)
      {
         info->FromHost = false; // Wifi.cpp knows which host it joined
         info->Kind = len > 0 ? rx.Kind : 0;
         info->Timestamp = len > 0 ? rx.Timestamp : 0;
      }
      return len;
   }

   int MP_WaitReplies(int count)
   {
      return TamaribaMP::WaitReplies(count);
   }

   bool MP_WaitHostFrame()
   {
      return TamaribaMP::WaitHostFrame();
   }

   bool LAN_Init()
   {
      return TamaribaLAN::Init();
   }

   void LAN_DeInit()
   {
      TamaribaLAN::DeInit();
   }

   int LAN_SendPacket(u8* data, int len)
   {
      return TamaribaLAN::SendPacket(data, len);
   }

   int LAN_RecvPacket(u8* data)
   {
      return TamaribaLAN::RecvPacket(data);
   }

#ifdef HAVE_OPENGL
   void* GL_GetProcAddress(const char* proc)
   {
      return (void*)nullptr;
   }
#endif

   void Sleep(u64 usecs)
   {
      retro_sleep(usecs / 1000);
   }
};
